"""
集中配置模块

所有外部依赖（大模型、Embedding/Rerank、Qdrant、MySQL、鉴权、上下文预算）
都从这里读取环境变量，避免配置散落在各个工具和 Agent 模块中。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# find_dotenv 会从本文件向上查找，支持 .env 放在项目根目录或上层工作目录
load_dotenv(find_dotenv(usecwd=False))

# 供应商别名映射：不同中转站对同一模型的命名不一致，这里做一层归一化
_MODEL_ALIASES = {
    "deepseek-v4.1-flash": "deepseek-flash",
    "deepseek-v4.1": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
}

# 硅基流动免费模型白名单（官方定价 ¥0）。
# 只允许这两个加入；任何 Pro/ 前缀或未确认免费的模型都会被配置校验拒绝。
FREE_EMBEDDING_MODELS = frozenset({"BAAI/bge-large-zh-v1.5", "BAAI/bge-m3"})
FREE_RERANK_MODELS = frozenset({"BAAI/bge-reranker-v2-m3"})


def _env(*names: str, default: str = "") -> str:
    """按优先级读取第一个非空环境变量。"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """运行时配置快照。"""

    # 对话模型（OpenAI 兼容协议）
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout: float
    # 单进程允许同时在途的模型请求数上限；中转站对并发敏感，必须显式限制
    llm_max_concurrency: int
    # SDK 重试之外的额外重试次数，用于覆盖中转站连续 5xx
    llm_extra_retries: int
    # 是否强制流式：部分中转即使非流式请求也返回 SSE，导致 SDK 解析崩溃
    llm_force_streaming: bool

    # 向量与重排（硅基流动）
    siliconflow_api_key: str
    siliconflow_base_url: str
    embedding_model: str
    rerank_model: str
    embedding_dim: int
    embedding_batch_size: int
    # siliconflow = 调用云端 API；local = 本地确定性哈希向量，仅供离线开发与联调
    embedding_provider: str
    rerank_provider: str

    # 向量库
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_timeout: float

    # 结构化业务库
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_charset: str
    mysql_collation: str
    mysql_sql_mode: str

    # 联网搜索
    tavily_api_key: str

    # 状态账本
    checkpoint_db_path: Path

    # 接口鉴权（为空表示本地开发不校验）
    access_token: str

    # 需要人工确认的高风险工具
    hitl_interrupt_tools: tuple[str, ...] = field(default_factory=tuple)

    # 上下文窗口预算（字符数）
    context_max_chars: int = 24000
    # 单次任务允许的模型调用步数上限，防止无限检索
    max_model_calls: int = 24

    @property
    def knowledge_base_dir(self) -> Path:
        return PROJECT_ROOT / "docs" / "knowledge_base"

    @property
    def output_dir(self) -> Path:
        return PROJECT_ROOT / "output"

    @property
    def upload_dir(self) -> Path:
        return PROJECT_ROOT / "updated"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取并缓存配置；测试中可通过 get_settings.cache_clear() 重新加载。"""

    llm_base_url = _env(
        "LLM_BASE_URL", "DEEPSEEK_BASE_URL", default="https://api.deepseek.com"
    ).rstrip("/")

    raw_model = _env("LLM_MODEL_NAME", "LLM_MODEL", "DEEPSEEK_MODEL", default="deepseek-flash")
    # 别名只在官方 DeepSeek 接口上成立：官方只认 deepseek-flash / deepseek-v4-pro，
    # 而中转站通常反过来只认 deepseek-v4.1-flash。按 base_url 区分，避免改写后 503。
    if "api.deepseek.com" in llm_base_url:
        model = _MODEL_ALIASES.get(raw_model, raw_model)
    else:
        model = raw_model
    checkpoint_path = Path(
        _env(
            "CHECKPOINT_DB_PATH",
            default=str(PROJECT_ROOT / "data" / "checkpoints.sqlite"),
        )
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    interrupt_tools = tuple(
        item.strip()
        for item in _env("HITL_INTERRUPT_TOOLS", default="convert_md_to_pdf").split(",")
        if item.strip()
    )

    embedding_model = _env("EMBEDDING_MODEL", default="BAAI/bge-large-zh-v1.5")
    rerank_model = _env("RERANK_MODEL", default="BAAI/bge-reranker-v2-m3")

    # 硅基流动免费额度约束：只允许使用定价为 ¥0 的向量与重排模型。
    # Pro/ 前缀为按量计费版本，云端的 bge 付费变体也不在白名单内。
    if embedding_model not in FREE_EMBEDDING_MODELS:
        raise ValueError(
            f"EMBEDDING_MODEL={embedding_model} 不在免费白名单中。"
            f"可用模型：{', '.join(sorted(FREE_EMBEDDING_MODELS))}"
        )
    if rerank_model not in FREE_RERANK_MODELS:
        raise ValueError(
            f"RERANK_MODEL={rerank_model} 不在免费白名单中。"
            f"可用模型：{', '.join(sorted(FREE_RERANK_MODELS))}"
        )

    return Settings(
        llm_api_key=_env("LLM_API_KEY", "DEEPSEEK_API_KEY"),
        llm_base_url=llm_base_url,
        llm_model=model,
        llm_timeout=_env_float("LLM_TIMEOUT_SECONDS", 120.0),
        llm_max_concurrency=max(1, _env_int("LLM_MAX_CONCURRENCY", 3)),
        llm_extra_retries=max(0, _env_int("LLM_EXTRA_RETRIES", 1)),
        llm_force_streaming=_env("LLM_FORCE_STREAMING", default="true").lower()
        not in {"0", "false", "no"},
        siliconflow_api_key=_env("SILICONFLOW_API_KEY"),
        siliconflow_base_url=_env(
            "SILICONFLOW_BASE_URL", default="https://api.siliconflow.cn/v1"
        ).rstrip("/"),
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
        embedding_provider=_env("EMBEDDING_PROVIDER", default="siliconflow").lower(),
        rerank_provider=_env("RERANK_PROVIDER", default="siliconflow").lower(),
        qdrant_url=_env("QDRANT_URL", default="http://localhost:6333"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        qdrant_collection=_env("QDRANT_COLLECTION", default="internal_kb"),
        qdrant_timeout=_env_float("QDRANT_TIMEOUT_SECONDS", 30.0),
        mysql_host=_env("MYSQL_HOST", default="localhost"),
        mysql_port=_env_int("MYSQL_PORT", 3306),
        mysql_user=_env("MYSQL_USER"),
        mysql_password=_env("MYSQL_PASSWORD"),
        mysql_database=_env("MYSQL_DATABASE", default="nexus_research_db"),
        mysql_charset=_env("MYSQL_CHARSET", default="utf8mb4"),
        mysql_collation=_env("MYSQL_COLLATION", default="utf8mb4_unicode_ci"),
        mysql_sql_mode=_env("MYSQL_SQL_MODE", default="TRADITIONAL"),
        tavily_api_key=_env("TAVILY_API_KEY"),
        checkpoint_db_path=checkpoint_path,
        access_token=_env("ACCESS_TOKEN"),
        hitl_interrupt_tools=interrupt_tools,
        context_max_chars=_env_int("CONTEXT_MAX_CHARS", 24000),
        max_model_calls=max(4, _env_int("MAX_MODEL_CALLS", 24)),
    )
