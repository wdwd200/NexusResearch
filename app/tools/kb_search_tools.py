"""
内部知识库检索工具

把企业研报、SOP 等私有文档切片后写入 Qdrant，
检索时先做向量粗召回，再用 BGE-Reranker 精排，只把 Top-N 证据片段交给子智能体。

对子智能体暴露两个工具：
- list_knowledge_domains：列出知识库覆盖的业务专题和文档清单；
- search_internal_knowledge：按专题和问题检索证据片段。
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from app.api.monitor import monitor
from app.rag import qdrant_store

# 精排分数下限，过滤明显不相关的片段；bge-reranker-v2-m3 的相关片段通常在 0.3 以上
_MIN_SCORE = 0.05


@tool
def list_knowledge_domains() -> str:
    """
    列出内部知识库包含的业务专题与文档清单

    作用：让子智能体先了解知识库的覆盖范围，再决定检索哪个专题。
    在任何检索动作之前都应先调用本工具，避免对不存在的专题做无效检索。
    :return: 中文格式的专题、文档名和片段数量说明
    """
    monitor.report_tool(tool_name="内部知识库专题列表工具：list_knowledge_domains")

    try:
        summary = qdrant_store.list_domains()
    except Exception as exc:
        return f"读取内部知识库失败，请确认 Qdrant 服务可用：{exc}"

    if not summary:
        return (
            "内部知识库当前为空。请先运行 scripts/ingest_knowledge_base.py "
            "把 docs/knowledge_base 下的文档写入 Qdrant。"
        )

    lines = ["内部知识库包含以下业务专题："]
    for domain, bucket in sorted(summary.items()):
        lines.append(f"\n【{domain}】共 {bucket['chunks']} 个片段，文档清单：")
        for source, count in sorted(bucket["sources"].items()):
            lines.append(f"  - {source}（{count} 个片段）")
    return "\n".join(lines)


@tool
def search_internal_knowledge(
    domain: Annotated[str, "业务专题名称，必须来自 list_knowledge_domains；不确定时传空字符串表示全库检索"],
    query: Annotated[str, "需要检索的具体问题或关键词"],
) -> str:
    """
    在内部私有知识库中检索与问题最相关的证据片段

    检索流程：向量粗召回 Top-15 -> BGE-Reranker 交叉重排 -> 返回 Top-5 片段。
    适合查询企业内部研报、行业报告、制度流程等非公开文档内容。
    :param domain: 业务专题；空字符串表示不限专题
    :param query: 检索问题
    :return: 带出处和相关性分数的证据片段文本
    """
    monitor.report_tool(
        tool_name="内部知识库检索工具：search_internal_knowledge",
        args={"domain": domain, "query": query},
    )

    try:
        results = qdrant_store.search_chunks(
            query=query,
            domain=domain.strip() or None,
            top_k=15,
            top_n=5,
            score_threshold=_MIN_SCORE,
        )
    except Exception as exc:
        return f"内部知识库检索失败，请确认 Qdrant 与硅基流动服务可用：{exc}"

    if not results:
        scope = f"专题「{domain}」" if domain.strip() else "全库"
        return f"{scope}中没有检索到与「{query}」相关的证据片段，请换一个角度或关键词再试。"

    lines = [f"检索到 {len(results)} 条相关证据（已按相关性排序）："]
    for index, item in enumerate(results, start=1):
        lines.append(
            f"\n[证据 {index}] 专题：{item['domain']} | 出处：{item['source']} "
            f"| 相关性：{item['score']}\n{item['text']}"
        )
    return "\n".join(lines)
