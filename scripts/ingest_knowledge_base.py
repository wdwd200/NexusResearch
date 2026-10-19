"""
内部知识库初始化脚本

把 docs/knowledge_base 下的 PDF/Markdown/TXT 文档按专题切片，
调用硅基流动 Embedding 向量化后写入 Qdrant。

用法：
    uv run python scripts/ingest_knowledge_base.py
    uv run python scripts/ingest_knowledge_base.py --source-dir docs/knowledge_base --reset

目录约定：一级子目录名即业务专题（domain），文件名即文档出处（source）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许直接以脚本方式运行：把项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.rag import qdrant_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def extract_pdf_text(path: Path) -> str:
    """逐页提取 PDF 文本；扫描版页面没有文本层时会跳过。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
        raise RuntimeError("缺少 pypdf，请先执行 uv sync") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # 单页解析失败不影响整份文档
            logger.warning("页面解析失败 %s: %s", path.name, exc)
    return "\n".join(pages)


def extract_text(path: Path) -> str:
    """按后缀分发到对应的文本提取实现。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"不支持的文件类型：{suffix}")


def collect_documents(source_dir: Path) -> list[tuple[str, Path]]:
    """
    扫描知识库目录。

    目录结构约定为 source_dir/<domain>/<file>；直接放在根目录的文件归入“通用”专题。
    """
    documents: list[tuple[str, Path]] = []
    if not source_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在：{source_dir}")

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(source_dir)
        domain = relative.parts[0] if len(relative.parts) > 1 else "通用"
        documents.append((domain, path))
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description="把内部知识库文档写入 Qdrant")
    parser.add_argument(
        "--source-dir",
        default=str(get_settings().knowledge_base_dir),
        help="知识库根目录，默认 docs/knowledge_base",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="每个片段的字符数，默认 400（bge 输入上限 512 token）",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=60,
        help="片段之间的重叠字符数，默认 60",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="入库前清空同名集合，适合全量重建",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = (PROJECT_ROOT / source_dir).resolve()

    documents = collect_documents(source_dir)
    if not documents:
        logger.error("在 %s 下没有找到可入库文档（支持 pdf/md/txt）", source_dir)
        return 1

    if args.reset:
        client = qdrant_store.get_qdrant_client()
        collection = get_settings().qdrant_collection
        if client.collection_exists(collection):
            logger.info("删除已有集合 %s", collection)
            client.delete_collection(collection)

    total = 0
    for domain, path in documents:
        logger.info("处理 %s/%s", domain, path.name)
        text = extract_text(path)
        if not text.strip():
            logger.warning("跳过无文本内容的文档（可能是扫描版 PDF）：%s", path.name)
            continue

        chunks = list(
            qdrant_store.iter_chunks_from_text(
                text,
                domain=domain,
                source=path.name,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        )
        if not chunks:
            logger.warning("文档切片为空，跳过：%s", path.name)
            continue

        written = qdrant_store.upsert_chunks(chunks)
        total += written
        logger.info("  -> %s 条片段", written)

    stats = qdrant_store.collection_stats()
    logger.info("入库完成，本次写入 %s 条，集合当前共 %s 条", total, stats["points"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
