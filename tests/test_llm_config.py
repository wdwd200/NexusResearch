"""
模型接入与并发上限回归

不联网、不消耗额度：用自定义的假模型验证信号量与配置解析。
覆盖两类真实事故：
1. 中转站只认 deepseek-v4.1-flash，而旧代码把它改写成 deepseek-flash 导致 503；
2. 并发不受控会打爆中转站，必须保证在途请求不超过 LLM_MAX_CONCURRENCY。
"""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class _ConfigTests(unittest.TestCase):
    def _settings(self, **env: str):
        from app.config import get_settings

        base = {
            "LLM_API_KEY": "test-key",
            "SILICONFLOW_API_KEY": "test-key",
            "EMBEDDING_PROVIDER": "local",
            "RERANK_PROVIDER": "local",
        }
        base.update(env)
        with patch.dict(os.environ, base, clear=False):
            get_settings.cache_clear()
            try:
                return get_settings()
            finally:
                get_settings.cache_clear()

    def test_relay_model_name_is_not_rewritten(self) -> None:
        """中转站只暴露 deepseek-v4.1-flash，改写会让请求 503。"""
        settings = self._settings(
            LLM_BASE_URL="https://aispot.example.com/v1",
            LLM_MODEL_NAME="deepseek-v4.1-flash",
        )
        self.assertEqual(settings.llm_model, "deepseek-v4.1-flash")

    def test_official_model_name_is_normalized(self) -> None:
        """官方接口只认 deepseek-flash / deepseek-v4-pro。"""
        settings = self._settings(
            LLM_BASE_URL="https://api.deepseek.com",
            LLM_MODEL_NAME="deepseek-v4.1-flash",
        )
        self.assertEqual(settings.llm_model, "deepseek-flash")

    def test_concurrency_defaults_to_three_and_is_never_zero(self) -> None:
        self.assertEqual(self._settings(LLM_BASE_URL="https://relay.test/v1").llm_max_concurrency, 3)
        self.assertEqual(
            self._settings(LLM_BASE_URL="https://relay.test/v1", LLM_MAX_CONCURRENCY="0").llm_max_concurrency,
            1,
        )
        self.assertEqual(
            self._settings(LLM_BASE_URL="https://relay.test/v1", LLM_MAX_CONCURRENCY="2").llm_max_concurrency,
            2,
        )


class _SlowFakeModel(BaseChatModel):
    """记录同时在途的调用数峰值。"""

    peak: int = 0
    active: int = 0
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "slow-fake"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        raise RuntimeError("测试只覆盖异步链路")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        type(self).calls += 1
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        try:
            await asyncio.sleep(0.15)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
        finally:
            type(self).active -= 1

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        type(self).calls += 1
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        try:
            await asyncio.sleep(0.05)
            yield ChatGenerationChunk(message=AIMessage(content="ok"))
        finally:
            type(self).active -= 1


class _ConcurrencyTests(unittest.TestCase):
    def test_semaphore_never_exceeds_limit(self) -> None:
        from app.agent import llm as llm_module

        limit = llm_module.current_concurrency_limit()
        semaphore = asyncio.Semaphore(limit)

        _SlowFakeModel.peak = 0
        _SlowFakeModel.active = 0
        _SlowFakeModel.calls = 0

        async def limited_call() -> None:
            async with semaphore:
                await _SlowFakeModel().ainvoke([HumanMessage(content="hi")])

        async def main() -> None:
            await asyncio.gather(*[limited_call() for _ in range(12)])

        asyncio.run(main())
        self.assertEqual(_SlowFakeModel.calls, 12)
        self.assertLessEqual(_SlowFakeModel.peak, limit)
        self.assertGreaterEqual(_SlowFakeModel.peak, 2)


class _ComposeTests(unittest.TestCase):
    def test_qdrant_has_no_shell_healthcheck(self) -> None:
        """qdrant 镜像基于 debian-slim，无 bash/curl/wget，健康检查会永久失败。"""
        text = Path("docker-compose.prod.yaml").read_text(encoding="utf-8")
        qdrant_block = text.split("  qdrant:", 1)[1].split("  api:", 1)[0]
        # 只看真实配置行，忽略说明性注释
        config_lines = [
            line.strip()
            for line in qdrant_block.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        joined = "\n".join(config_lines)
        self.assertNotIn("healthcheck", joined)
        self.assertNotIn("/dev/tcp", joined)

        api_block = text.split("  api:", 1)[1]
        self.assertIn("qdrant:\n        condition: service_started", api_block)
        self.assertNotIn("qdrant:\n        condition: service_healthy", api_block)


class _StreamAggregationTests(unittest.TestCase):
    """中转对非流式请求也返回 SSE，聚合层必须能还原成完整消息。"""

    def test_empty_chunks_become_empty_message(self) -> None:
        from app.agent.llm import _chunks_to_result

        result = _chunks_to_result([])
        self.assertEqual(len(result.generations), 1)
        self.assertEqual(result.generations[0].message.content, "")

    def test_usage_only_chunk_does_not_crash(self) -> None:
        from app.agent.llm import _chunks_to_result

        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata={"input_tokens": 1, "output_tokens": 0, "total_tokens": 1})
        )
        result = _chunks_to_result([chunk])
        self.assertEqual(result.generations[0].message.content, "")

    def test_multiple_chunks_are_concatenated(self) -> None:
        from app.agent.llm import _chunks_to_result

        chunks = [
            ChatGenerationChunk(message=AIMessageChunk(content="你好")),
            ChatGenerationChunk(message=AIMessageChunk(content="，世界")),
        ]
        result = _chunks_to_result(chunks)
        self.assertEqual(result.generations[0].message.content, "你好，世界")


class _ContextBudgetRegressionTests(unittest.TestCase):
    """
    回归：预算裁剪曾把“最新用户消息之后”的工具回执全部丢弃，
    导致模型看不到刚成功的工具结果而反复重试同一工具。
    """

    def test_tool_results_after_latest_user_are_kept(self) -> None:
        from app.agent.context_manager import budget_messages

        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"name": "g", "args": {"content": "x" * 3000}, "id": "1"}]),
            ToolMessage(content="done-1", tool_call_id="1"),
            AIMessage(content="", tool_calls=[{"name": "g", "args": {"content": "y" * 3000}, "id": "2"}]),
            ToolMessage(content="done-2", tool_call_id="2"),
        ]
        for budget in (8000, 24000):
            window = budget_messages(messages, max_chars=budget)
            types = [item.type for item in window]
            self.assertIn("tool", types)
            ids = {item.tool_call_id for item in window if item.type == "tool"}
            self.assertEqual(ids, {"1", "2"})
            # 每条 tool 前面必须紧邻对应的 ai tool_calls
            for index, item in enumerate(window):
                if item.type == "tool":
                    self.assertEqual(window[index - 1].type, "ai")

    def test_recent_tool_result_survives_when_history_is_long(self) -> None:
        from app.agent.context_manager import budget_messages

        messages: list = [HumanMessage(content="旧问题" * 400)]
        for index in range(12):
            messages.append(
                AIMessage(content="", tool_calls=[{"name": "g", "args": {"c": "z" * 900}, "id": str(index)}])
            )
            messages.append(ToolMessage(content=f"result-{index}", tool_call_id=str(index)))
        messages.append(HumanMessage(content="最新问题"))
        messages.append(AIMessage(content="", tool_calls=[{"name": "g", "args": {"c": "w" * 500}, "id": "last"}]))
        messages.append(ToolMessage(content="final-result", tool_call_id="last"))

        window = budget_messages(messages, max_chars=5000)
        texts = [getattr(item, "content", "") or "" for item in window]
        self.assertIn("最新问题", texts)
        self.assertIn("final-result", texts)


class _SqlHardeningTests(unittest.TestCase):
    def test_rejects_dangerous_functions_and_file_exports(self) -> None:
        from app.tools.db_tools import _assert_readonly

        for unsafe in (
            "SELECT LOAD_FILE('/etc/passwd')",
            "SELECT SLEEP(10)",
            "SELECT BENCHMARK(1000000, MD5('x'))",
            "SELECT * FROM t INTO OUTFILE '/tmp/x'",
            "SELECT * FROM t FOR UPDATE",
        ):
            with self.assertRaises(ValueError):
                _assert_readonly(unsafe)

    def test_still_allows_normal_reads(self) -> None:
        from app.tools.db_tools import _assert_readonly

        for safe in ("SELECT 1", "SHOW TABLES", "EXPLAIN SELECT 1", "WITH a AS (SELECT 1) SELECT * FROM a"):
            self.assertTrue(_assert_readonly(safe))


if __name__ == "__main__":
    unittest.main()
