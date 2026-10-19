"""
MySQL 数据库查询工具模块

封装数据库查询助手使用的三个 LangChain 工具：
list_sql_tables 用于发现真实表名，get_table_data 用于预览字段和样例数据，
execute_sql_query 用于在确认结构后执行自定义查询。
"""

import re

import sqlglot
from langchain_core.tools import tool
from mysql.connector import Error, connect
from app.api.monitor import monitor
from app.config import get_settings

# 只读语句类型名。写操作、多语句、存储过程调用一律拒绝。
# 数据库只读账号是第二层；这里不修改真实库账号或权限。
_READONLY_KEYS = {"select", "union", "show", "describe", "explain"}
_FORBIDDEN_KEYS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "command",
    "merge",
    "copy",
    "transaction",
    "commit",
    "rollback",
    "set",
    "kill",
    "grant",
    "revoke",
}

# 危险函数/子句：即便语句是 SELECT，这些也会越出“只读数据查询”的边界。
# - LOAD_FILE / INTO OUTFILE / INTO DUMPFILE：读写 MySQL 服务端文件系统；
# - SLEEP / BENCHMARK：可被用来拖垮数据库连接；
# - INFORMATION_SCHEMA 不拦，业务探索需要它。
_FORBIDDEN_FUNCTIONS = {"load_file", "sleep", "benchmark", "get_lock", "release_lock"}
_FORBIDDEN_SQL_FRAGMENTS = ("into outfile", "into dumpfile", "for update", "lock in share mode")


def _assert_readonly(sql: str) -> str:
    """
    用 sqlglot 校验单条只读语句。

    开头正则挡不住 SELECT 后面拼接的写语句，也不能识别注释包裹的多语句。
    解析失败、出现写节点或危险函数时直接拒绝，不把原文交给数据库。
    """
    text = str(sql or "").strip()
    if not text:
        raise ValueError("SQL 为空")
    if "\x00" in text:
        raise ValueError("SQL 包含非法字符")

    lowered = text.lower()
    for fragment in _FORBIDDEN_SQL_FRAGMENTS:
        if fragment in lowered:
            raise ValueError("只允许只读查询，禁止导出文件或加锁读")

    try:
        statements = sqlglot.parse(text, read="mysql")
    except sqlglot.errors.ParseError as exc:
        raise ValueError("SQL 无法解析，已拒绝执行") from exc

    statements = [item for item in statements if item is not None]
    if len(statements) != 1:
        raise ValueError("只允许单条只读 SQL")

    statement = statements[0]
    keys = {item.key for item in statement.walk()}
    if keys & _FORBIDDEN_KEYS:
        raise ValueError("只允许只读查询（SELECT / SHOW / DESCRIBE / EXPLAIN）")
    if statement.key not in _READONLY_KEYS:
        raise ValueError("只允许只读查询（SELECT / SHOW / DESCRIBE / EXPLAIN）")

    # 逐个检查函数调用节点，拦住 LOAD_FILE/SLEEP 这类“看起来是 SELECT”的危险函数
    for node in statement.walk():
        if isinstance(node, sqlglot.exp.Func):
            if str(node.name or "").lower() in _FORBIDDEN_FUNCTIONS:
                raise ValueError(f"禁止调用危险函数：{node.name}")
    return statement.sql(dialect="mysql")


# 集中读取数据库配置，后续三个工具都复用这份连接参数
def get_db_config():
    """
    从集中配置读取 MySQL 连接参数

    所有数据库工具都通过此函数拿到同一份连接参数，避免每个工具重复读取环境变量
    :return: mysql.connector.connect 可直接使用的连接参数
    """
    settings = get_settings()
    config = {
        "host": settings.mysql_host,
        "port": settings.mysql_port,
        "user": settings.mysql_user,
        "password": settings.mysql_password,
        "database": settings.mysql_database,
        "charset": settings.mysql_charset,
        "collation": settings.mysql_collation,
        "autocommit": True,
        "sql_mode": settings.mysql_sql_mode,
    }

    # 去掉未配置的可选项，避免把 None 传给 mysql.connector 造成连接参数异常
    config = {k: v for k, v in config.items() if v is not None}

    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"缺失数据库核心配置：{', '.join(missing_keys)}")

    return config


@tool
def list_sql_tables() -> str:
    """
    查询当前数据库中所有可用表

    作用：让模型先识别真实可用的表名，方便后续预览表结构和编写自定义 SQL。
    :return: 有表：可用的表有：表1,表2,表3...
             没有表：没有可用的表
             出现异常：查询出现异常：异常信息
    """

    # 埋点：工具一被调用，前端可以展示当前正在查询数据库表名
    monitor.report_tool(tool_name="数据库表名查询工具：list_sql_tables", args={})

    # 加载数据库连接信息
    config = get_db_config()

    # MySQL 查询的固定步骤：
    # 1. 创建连接
    # 2. 创建 cursor
    # 3. 执行 SQL
    # 4. 获取返回结果
    # 5. 释放连接和 cursor 资源
    # 这里捕获异常并返回中文提示，避免工具报错直接中断 Agent 执行链路
    try:
        # 使用 with 管理连接和游标，查询结束后自动释放数据库资源
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                sql = "SHOW TABLES"
                cursor.execute(sql)

                # SHOW TABLES 返回形如：[("drugs",), ("inventory",), ("sales_records",)]
                tables = cursor.fetchall()
                if not tables:
                    return "没有可用的表"

                # 取每个元组的第一个元素，拼成模型容易阅读的表名列表
                table_names = [table[0] for table in tables]
                return f"可用的表有：{', '.join(table_names)}"
    except Error as e:
        return f"查询出现异常：{e!s}"


@tool
def get_table_data(table_name) -> str:
    """
    查询指定表的前 100 行数据

    当前工具调用之前，应先调用 list_sql_tables 完成表名校验。
    此工具的作用：
    1. 完成单表样例数据查询
    2. 为多表查询提供表结构信息和数据格式参考
    :param table_name: 表名
    :return: CSV 格式数据
             1. 第一行是列信息，列之间使用英文逗号分隔
             2. 第二行开始是表数据，值之间也使用英文逗号分隔
             3. 行和行之间使用 \n 分隔
             4. 至多查询 100 条表数据
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n
                1,张三,18\n -> 至多查询 100 条
    """
    # 埋点：工具二被调用，前端可以展示当前正在预览哪张表
    monitor.report_tool(
        tool_name="数据库表数据查询工具：get_table_data",
        args={"table_name": table_name},
    )

    # 获取数据库参数
    config = get_db_config()

    # 查询流程同样是：连接 -> cursor -> 执行 SQL -> 获取列信息和数据 -> 自动释放资源
    # 只允许查询单表且表名必须是合法标识符，避免拼接任意 SQL
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(table_name)):
        return "表名不合法：只允许字母、数字和下划线。"
    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                sql = f"SELECT * FROM `{table_name}` LIMIT 100"
                cursor.execute(sql)

                # cursor.description 保存查询结果的列元信息
                # 例如：[("id", ...), ("name", ...), ("age", ...)]
                # 如果 SQL 没有结果集，description 可能为 None
                description = cursor.description
                if not description:
                    return f"数据表 {table_name} 暂无数据。"

                # 只取每个列信息元组的第一个元素，也就是列名
                # 例如：["id", "name", "age"]
                columns = [desc[0] for desc in description]

                # fetchall 返回表数据，形如：[(1, "张三", 18), (2, "李四", 20)]
                rows = cursor.fetchall()

                # 把每一行数据从元组转成 CSV 行文本
                # 例如：(1, "张三", 18) -> "1,张三,18"
                results = [",".join(map(str, row)) for row in rows]

                # columns 组成 CSV 头部，rows 组成 CSV 数据体
                # 最终返回：
                # id,name,age
                # 1,张三,18
                header_str = ",".join(columns)
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except Error as e:
        return f"查询出现异常：{e!s}"


@tool
def execute_sql_query(query) -> str:
    """
    执行自定义 SQL 查询

    切记：执行之前，需要通过 list_sql_tables 明确真实表名，
    再通过 get_table_data 明确表结构和数据格式。
    适合多表关联、筛选、聚合、排序等复杂查询。
    :param query: 要执行的自定义 SQL 语句
    :return: CSV 格式数据
             1. 第一行是列信息，列之间使用英文逗号分隔
             2. 第二行开始是表数据，值之间也使用英文逗号分隔
             3. 行和行之间使用 \n 分隔
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n
    """
    # 埋点：记录模型最终生成的 SQL，便于排查是否命中正确表字段
    monitor.report_tool(
        tool_name="数据库表数据查询工具：execute_sql_query", args={"query": query}
    )

    # 获取数据库参数
    config = get_db_config()

    # 自定义查询和 get_table_data 的结果处理逻辑一致：
    # 执行 SQL -> 读取 description 得到列名 -> fetchall 得到数据 -> 拼成 CSV 返回
    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                safe_sql = _assert_readonly(query)
                cursor.execute(safe_sql)

                # 非查询类 SQL 没有结果集描述，这里统一返回提示，避免工具调用直接抛错给模型
                description = cursor.description
                if not description:
                    return f"执行自定义 SQL 语句没有查询结果，SQL 为：{query}"
                # description => [("列1", ...), ("列2", ...)]
                columns = [desc[0] for desc in description]

                # rows => [(值1, 值2), (值1, 值2)]
                rows = cursor.fetchall()

                # 每行元组统一转为逗号分隔文本，便于模型读取和后续整理
                results = [",".join(map(str, row)) for row in rows]

                # 第一行是列名，后续是查询数据
                header_str = ",".join(columns)
                data_str = "\n".join(results)
                return f"{header_str}\n{data_str}"
    except Error as e:
        return f"查询出现异常：{e!s}"


if __name__ == "__main__":
    # 本地调试入口：直接运行本文件可验证 .env 中的 MySQL 连接配置是否可用
    print(
        execute_sql_query.invoke(
            {
                "query": "SELECT * FROM `drugs` dgs join sales_records srd on dgs.drug_id = srd.drug_id"
            }
        )
    )
