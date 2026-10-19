"""
大模型初始化模块

统一通过 OpenAI 兼容协议接入对话模型，主智能体和子智能体共享同一个实例。
模型名、地址、密钥和并发上限统一由 app.config 读取。

并发控制是硬约束：中转站对同时进行的请求数敏感，超限会影响同站其他用户。
这里用进程级 asyncio.Semaphore 限流，并给 httpx 客户端设置同规模连接池，
从两层保证在途请求数不超过 LLM_MAX_CONCURRENCY（默认 3）。

注意覆盖范围：LangChain 的普通调用走 `_generate/_agenerate`，流式调用走
`_stream/_astream`，四者不互相复用，必须分别加锁，否则会绕过限流。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# 进程级并发闸门。所有模型调用都要先拿到这个信号量。
_llm_semaphore = asyncio.Semaphore(settings.llm_max_concurrency)

# 连接池不放大并发：max_connections 与信号量上限一致，避免排队请求堆积在连接层。
_http_client = httpx.Client(
    timeout=httpx.Timeout(settings.llm_timeout, connect=15.0),
    limits=httpx.Limits(
        max_connections=settings.llm_max_concurrency,
        max_keepalive_connections=settings.llm_max_concurrency,
    ),
)
_http_async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(settings.llm_timeout, connect=15.0),
    limits=httpx.Limits(
        max_connections=settings.llm_max_concurrency,
        max_keepalive_connections=settings.llm_max_concurrency,
    ),
)


_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _is_transient(exc: BaseException) -> bool:
    """中转站的 5xx/429 属于瞬时故障，值得退避重试；4xx 参数错误重试也没用。"""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    return status in _TRANSIENT_STATUS


def _chunks_to_result(chunks: list[ChatGenerationChunk]) -> ChatResult:
    """
    把流式片段聚合成一个 ChatResult。

    中转站偶发返回空 choices 的 SSE（只有 usage），此时聚合结果是空消息，
    让上层按正常空回复处理，而不是在 SDK 内部抛 AttributeError。
    """
    if not chunks:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=""))]
        )

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk

    generation_info = dict(merged.generation_info or {})
    message = merged.message
    usage = getattr(message, "usage_metadata", None)
    return ChatResult(
        generations=[
            ChatGeneration(message=message, generation_info=generation_info or None)
        ],
        llm_output={"token_usage": usage} if usage else None,
    )


class ConcurrencyLimitedChatModel(ChatOpenAI):
    """
    在模型调用外层套一层信号量的 ChatOpenAI。

    四个生成入口都覆盖：同步、异步、同步流式、异步流式。
    本项目主链路走异步流式，同步入口主要服务于脚本和调试。

    除 OpenAI SDK 自带重试外，这里再叠一层带指数退避的重试。
    原因是中转站会出现连续 502，SDK 的 2 次重试不足以覆盖。
    每层重试都在信号量内部，不会放大在途并发。
    """

    # 额外重试次数（不含 SDK 自身重试）
    extra_retries: int = 1
    retry_backoff_seconds: float = 2.0

    @property
    def llm_type(self) -> str:
        return "concurrency-limited-openai"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError(
            "同步模型调用未启用限流，请使用 ainvoke/astream 等异步入口"
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise RuntimeError(
            "同步流式调用未启用限流，请使用 astream 异步入口"
        )

    async def _call_with_retry(self, call):
        """带退避的异步调用包装，信号量由调用方持有。"""
        last_error: BaseException | None = None
        for attempt in range(self.extra_retries + 1):
            try:
                return await call()
            except Exception as exc:
                last_error = exc
                if not _is_transient(exc) or attempt >= self.extra_retries:
                    raise
                delay = self.retry_backoff_seconds * (attempt + 1)
                logger.warning(
                    "模型调用遇到瞬时故障（%s），%.1fs 后重试 %d/%d",
                    getattr(exc, "status_code", type(exc).__name__),
                    delay,
                    attempt + 1,
                    self.extra_retries,
                )
                await asyncio.sleep(delay)
        raise last_error  # pragma: no cover - 循环内已处理

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        async with _llm_semaphore:
            if not settings.llm_force_streaming:
                return await self._call_with_retry(
                    lambda: super(ConcurrencyLimitedChatModel, self)._agenerate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                )

            # 强制流式：部分中转对非流式请求也返回 SSE，SDK 解析会拿到 str 并崩在
            # response.model_dump()。这里改走流式再聚合成一条完整消息。
            collected: list[ChatGenerationChunk] = []
            async for chunk in self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                collected.append(chunk)
            return _chunks_to_result(collected)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # 信号量必须覆盖整个流的消费周期，不能在拿到首个 chunk 后释放，
        # 否则多个流会同时处于在途状态。
        async with _llm_semaphore:
            # 流式只在“尚未产出任何 chunk”时才可安全重试，否则会重复输出
            emitted = False
            attempt = 0
            while True:
                try:
                    async for chunk in super()._astream(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    ):
                        emitted = True
                        yield chunk
                    return
                except Exception as exc:
                    if emitted or not _is_transient(exc) or attempt >= self.extra_retries:
                        raise
                    attempt += 1
                    delay = self.retry_backoff_seconds * attempt
                    logger.warning(
                        "流式模型调用遇到瞬时故障（%s），%.1fs 后重试 %d/%d",
                        getattr(exc, "status_code", type(exc).__name__),
                        delay,
                        attempt,
                        self.extra_retries,
                    )
                    await asyncio.sleep(delay)


def build_model() -> BaseChatModel:
    """构造模型实例。缺少密钥时仍允许导入，真正调用时由接口返回 401。"""
    return ConcurrencyLimitedChatModel(
        model=settings.llm_model,
        api_key=settings.llm_api_key or "not-configured",
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=settings.llm_timeout,
        # 中转站偶发 502/503，重试 2 次可覆盖大部分瞬时故障；上限受并发闸门约束
        max_retries=2,
        http_client=_http_client,
        http_async_client=_http_async_client,
        extra_retries=max(0, settings.llm_extra_retries),
    )


model = build_model()


def current_concurrency_limit() -> int:
    """暴露给健康检查和测试，确认实际生效的上限。"""
    return settings.llm_max_concurrency
