"""
硅基流动向量与重排客户端

封装两条外部能力，供知识库工具和入库脚本共用：
- embed_texts：调用 OpenAI 兼容 /v1/embeddings，默认 BAAI/bge-large-zh-v1.5（1024 维）；
- rerank：调用 /v1/rerank，默认 BAAI/bge-reranker-v2-m3，对粗召回结果精排。

同时支持 EMBEDDING_PROVIDER=local / RERANK_PROVIDER=local 的离线模式：
用确定性哈希向量和词面重合度打分，保证没有外部额度时整条链路仍可联调。
生产环境使用默认的 siliconflow 即可。
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# check_embedding_ctx_length=False 让 SDK 直接发送原始字符串，
# 避免它用 tiktoken 预估 token 后再发 token id 数组（硅基流动不接受那种格式）
_embeddings_client = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.siliconflow_api_key or "not-configured",
    base_url=settings.siliconflow_base_url,
    check_embedding_ctx_length=False,
    chunk_size=settings.embedding_batch_size,
)


def _local_embed(text: str, dim: int) -> list[float]:
    """
    生成确定性哈希向量（离线模式专用）。

    用字符 n-gram 的哈希投影模拟词袋向量，同义改写无法匹配，
    因此只用于开发联调和流水线自测，不能替代真实语义模型。
    """
    vector = [0.0] * dim
    normalized = re.sub(r"\s+", "", text)
    grams = [normalized[i : i + 2] for i in range(max(len(normalized) - 1, 1))]
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _local_rerank(query: str, documents: Sequence[str], top_n: int) -> list[dict[str, Any]]:
    """离线重排：按查询与文档的字符重合度打分。"""
    query_chars = set(re.sub(r"\s+", "", query))

    scored: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        doc_chars = set(re.sub(r"\s+", "", document))
        if not doc_chars:
            score = 0.0
        else:
            overlap = len(query_chars & doc_chars)
            score = overlap / math.sqrt(len(doc_chars) + 1)
        scored.append({"index": index, "score": round(score, 4), "document": document})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_n]


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """
    批量向量化文本。

    :param texts: 待向量化文本列表
    :return: 与输入顺序一致的向量列表
    """
    if not texts:
        return []

    cleaned = [text if text and text.strip() else " " for text in texts]

    if settings.embedding_provider == "local":
        vectors = [_local_embed(text, settings.embedding_dim) for text in cleaned]
    else:
        vectors = _embeddings_client.embed_documents(cleaned)

    for index, vector in enumerate(vectors):
        if len(vector) != settings.embedding_dim:
            raise ValueError(
                f"Embedding 维度不一致：第 {index} 条返回 {len(vector)} 维，"
                f"配置期望 {settings.embedding_dim} 维。请检查 EMBEDDING_MODEL 与 Qdrant 集合维度。"
            )
    return vectors


def embed_query(text: str) -> list[float]:
    """向量化单个查询文本。"""
    return embed_texts([text])[0]


def rerank(
    query: str,
    documents: Sequence[str],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    对候选文档做交叉注意力精排。

    :param query: 用户查询
    :param documents: 候选文档片段
    :param top_n: 返回条数
    :return: 形如 [{"index": int, "score": float, "document": str}]，按分数降序
    """
    if not documents:
        return []

    if settings.rerank_provider == "local":
        return _local_rerank(query, documents, top_n)

    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": list(documents),
        "top_n": min(top_n, len(documents)),
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.siliconflow_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=settings.llm_timeout) as client:
            response = client.post(
                f"{settings.siliconflow_base_url}/rerank",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        # Rerank 失败时降级为向量相似度顺序，保证检索链路不中断
        logger.warning("Rerank 调用失败，降级为向量召回顺序：%s", exc)
        return [
            {"index": index, "score": 0.0, "document": document}
            for index, document in enumerate(documents[:top_n])
        ]

    results: list[dict[str, Any]] = []
    for item in data.get("results", []):
        index = int(item.get("index", 0))
        results.append(
            {
                "index": index,
                "score": float(item.get("relevance_score", 0.0)),
                "document": documents[index] if 0 <= index < len(documents) else "",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_n]
