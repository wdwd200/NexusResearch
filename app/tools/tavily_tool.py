"""
Tavily 网络搜索工具模块

封装 internet_search 工具，供网络搜索子智能体检索互联网公开信息。
客户端在使用时创建，避免缺少密钥时模块导入阶段直接抛异常。
"""

import os
from typing import Literal

from langchain_core.tools import tool
from tavily import TavilyClient

from app.api.monitor import monitor
from app.config import get_settings


# @tool 会把函数签名和 docstring 暴露给 DeepAgents，模型据此决定是否调用以及如何填参
@tool
def internet_search(
    query: str,
    topic: Literal["news", "finance", "general"] = "general",
    max_results: int = 5,
    include_raw_content: bool = False,
):
    """
    根据用户问题检索互联网公开信息

    注意：本工具只用于外部公开网页、新闻、政策等信息，不用于查询业务数据库或内部私有知识库
    :param query: 搜索关键词或自然语言问题
    :param topic: 搜索主题，可选 news、finance、general
    :param max_results: 返回的最大结果数
    :param include_raw_content: 是否返回网页原文内容；False 返回摘要，True 尝试返回更完整正文
    :return: Tavily 返回的结构化搜索结果
    """
    monitor.report_tool(
        tool_name="网络搜索工具",
        args={
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
        },
    )

    api_key = get_settings().tavily_api_key or os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        # 明确告诉模型“重试无意义”，否则它会对同一个失败工具反复调用直到耗尽预算
        return (
            "联网搜索不可用：未配置 TAVILY_API_KEY。"
            "这是永久性配置缺失，重复调用本工具不会成功，请立即改用其他信息源或直接基于已有信息作答。"
        )

    client = TavilyClient(api_key=api_key)
    try:
        return client.search(
            query=query,
            topic=topic,
            max_results=max_results,
            include_raw_content=include_raw_content,
        )
    except Exception as exc:
        return (
            f"联网搜索失败：{exc}。"
            "如为额度或网络问题，重试同一查询大概率仍失败，请换角度或改用其他信息源。"
        )


if __name__ == "__main__":
    from pprint import pprint

    pprint(internet_search.invoke({"query": "2026 年人工智能行业趋势"}))
