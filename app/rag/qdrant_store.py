"""
Qdrant 知识库存储层

负责内部知识库集合的建表、写入和检索。集合只存向量、原文片段和少量元数据，
不依赖任何外部重型 RAG 服务，单进程内存占用可以忽略。

设计要点：
- 单集合 + domain 负载字段做业务专题过滤，避免为每个专题单独建集合；
- 写入用确定性 point id，重复入库时覆盖而不是产生重复片段；
- 检索先按向量粗召回 Top-K，再交给 Rerank 精排 Top-N。
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.rag.embedding_client import embed_query, embed_texts, rerank

logger = logging.getLogger(__name__)

settings = get_settings()

# 用 uuid5 生成稳定 id：同一文档同一片段重复入库会覆盖，不会产生重复向量
_NAMESPACE = uuid.UUID("6f1d6f0e-6a9a-4e2c-9d3f-2a7b6c4d8e10")


@dataclass
class KnowledgeChunk:
    """一条待入库的知识片段。"""

    domain: str
    source: str
    chunk_index: int
    text: str

    @property
    def point_id(self) -> str:
        raw = f"{self.domain}|{self.source}|{self.chunk_index}|{self.text[:64]}"
        return str(uuid.uuid5(_NAMESPACE, hashlib.sha256(raw.encode("utf-8")).hexdigest()))


_client: QdrantClient | None = None
_client_key: str | None = None


def get_qdrant_client() -> QdrantClient:
    """
    获取 Qdrant 客户端（进程内单例）。

    QDRANT_URL 支持两种形式：
    - http(s)://host:port：连接独立 Qdrant 服务（生产默认）
    - 本地目录路径（或 local:/path）：使用 qdrant-client 嵌入式模式，便于无容器开发

    嵌入式模式不允许同一目录被多个客户端实例同时打开，所以必须复用单例；
    server 模式复用连接也能减少握手开销。
    """
    global _client, _client_key

    url = settings.qdrant_url
    api_key = settings.qdrant_api_key or None
    key = f"{url}|{api_key}|{settings.qdrant_timeout}"

    if _client is not None and _client_key == key:
        return _client

    if url.startswith("local:"):
        _client = QdrantClient(path=url[len("local:") :])
    elif "://" not in url:
        # 没有协议前缀时按本地目录处理，例如 QDRANT_URL=./data/qdrant
        _client = QdrantClient(path=url)
    else:
        _client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=settings.qdrant_timeout,
        )

    _client_key = key
    return _client


def reset_client_cache() -> None:
    """关闭并丢弃缓存的客户端，供测试或切换存储位置时使用。"""
    global _client, _client_key
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _client_key = None


def ensure_collection(client: QdrantClient | None = None) -> None:
    """确保知识库集合存在，维度与 Embedding 模型保持一致。"""
    client = client or get_qdrant_client()
    collection = settings.qdrant_collection

    if client.collection_exists(collection):
        info = client.get_collection(collection)
        size = info.config.params.vectors.size  # type: ignore[union-attr]
        if size != settings.embedding_dim:
            raise ValueError(
                f"Qdrant 集合 {collection} 现有维度 {size}，与配置维度 "
                f"{settings.embedding_dim} 不一致。请重建集合或修改 EMBEDDING_DIM。"
            )
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=settings.embedding_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    # domain 用于按业务专题过滤；source 用于定位文档出处
    client.create_payload_index(
        collection_name=collection,
        field_name="domain",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    logger.info("已创建 Qdrant 集合 %s（维度 %s）", collection, settings.embedding_dim)


def upsert_chunks(chunks: Sequence[KnowledgeChunk], batch_size: int | None = None) -> int:
    """
    批量写入知识片段。

    :param chunks: 待写入片段
    :param batch_size: 每批向量化并写入的数量
    :return: 写入条数
    """
    if not chunks:
        return 0

    client = get_qdrant_client()
    ensure_collection(client)
    batch = batch_size or settings.embedding_batch_size
    written = 0

    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        vectors = embed_texts([chunk.text for chunk in window])
        points = [
            qmodels.PointStruct(
                id=chunk.point_id,
                vector=vector,
                payload={
                    "domain": chunk.domain,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(window, vectors)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        written += len(points)

    logger.info("已写入 %s 条知识片段到 %s", written, settings.qdrant_collection)
    return written


def list_domains(client: QdrantClient | None = None) -> dict[str, dict[str, int]]:
    """
    汇总知识库中的业务专题与文档清单。

    :return: {domain: {"chunks": n, "sources": {source: chunk_count}}}
    """
    client = client or get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection):
        return {}

    summary: dict[str, dict[str, int]] = {}
    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=512,
            offset=next_offset,
            with_payload=["domain", "source"],
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            domain = str(payload.get("domain", "未分类"))
            source = str(payload.get("source", "未知文档"))
            bucket = summary.setdefault(domain, {"chunks": 0, "sources": {}})
            bucket["chunks"] += 1
            bucket["sources"][source] = bucket["sources"].get(source, 0) + 1
        if next_offset is None:
            break

    return summary


def _build_filter(domain: str | None) -> qmodels.Filter | None:
    if not domain:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="domain",
                match=qmodels.MatchValue(value=domain),
            )
        ]
    )


def search_chunks(
    query: str,
    domain: str | None = None,
    top_k: int = 15,
    top_n: int = 5,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    两阶段检索：向量粗召回 Top-K，再 Rerank 精排 Top-N。

    :param query: 检索问题
    :param domain: 限定业务专题；None 表示全库
    :param top_k: 粗召回条数
    :param top_n: 精排后返回条数
    :param score_threshold: 精排分数下限，低于该值的片段丢弃
    :return: [{"text", "source", "domain", "score"}]
    """
    if not query or not query.strip():
        return []

    client = get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection):
        return []

    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=embed_query(query),
        limit=top_k,
        query_filter=_build_filter(domain),
        with_payload=True,
    ).points

    if not hits:
        return []

    documents = [str((hit.payload or {}).get("text", "")) for hit in hits]
    ranked = rerank(query, documents, top_n=top_n)

    results: list[dict] = []
    for item in ranked:
        if score_threshold is not None and item["score"] < score_threshold:
            continue
        hit = hits[item["index"]]
        payload = hit.payload or {}
        results.append(
            {
                "text": documents[item["index"]],
                "source": str(payload.get("source", "")),
                "domain": str(payload.get("domain", "")),
                "score": round(item["score"], 4),
            }
        )
    return results


def collection_stats(client: QdrantClient | None = None) -> dict[str, int]:
    """返回集合规模，供健康检查和脚本输出。"""
    client = client or get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection):
        return {"points": 0}
    return {"points": client.count(settings.qdrant_collection, exact=True).count}


def iter_chunks_from_text(
    text: str,
    domain: str,
    source: str,
    chunk_size: int = 400,
    overlap: int = 60,
) -> Iterable[KnowledgeChunk]:
    """
    对纯文本做定长切片，保留少量重叠以避免语义被切断。

    默认 400 字符：bge-large-zh-v1.5 的输入上限是 512 token，中文近似 1 字 1 token，
    超过上限硅基流动会直接返回 400，因此必须留出余量。

    这里不用 LangChain 的 splitter，是因为知识库以中文报告为主，
    按字符长度切片已经足够稳定，也减少一层依赖。
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return

    index = 0
    position = 0
    while position < len(cleaned):
        end = min(position + chunk_size, len(cleaned))
        piece = cleaned[position:end].strip()
        if len(piece) >= 40:
            yield KnowledgeChunk(
                domain=domain,
                source=source,
                chunk_index=index,
                text=piece,
            )
            index += 1
        if end >= len(cleaned):
            break
        position = max(end - overlap, position + 1)
