"""有界本地验收：只用提示内事实生成 Markdown，审批后转 PDF。

不联网、不查业务库、不调用 Embedding。使用独立 checkpoint 和会话目录。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["CHECKPOINT_DB_PATH"] = str(Path(tempfile.gettempdir()) / "nexus-smoke-checkpoints.sqlite")
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["RERANK_PROVIDER"] = "local"
os.environ["HITL_INTERRUPT_TOOLS"] = "convert_md_to_pdf"

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.context_manager import ContextWindowMiddleware
from app.agent.llm import model
from app.agent.main_agent import get_pending_approval, project_root_path
from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.config import get_settings
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.utils.path_utils import session_directory
from deepagents import create_deep_agent

THREAD_ID = "smoke-local-report"
MAX_MODEL_CALLS = 8
RECURSION_LIMIT = 40
TIMEOUT_SECONDS = 110

FACTS = """
只用下面 3 条合成事实，不要补充外部信息：
1. 样例公司甲 2025 年公开营收为 100。
2. 样例公司乙 2025 年公开营收为 80。
3. 两者差额为 20。
只调用一次 generate_markdown，文件名 report.md。
写完后立刻调用 convert_md_to_pdf，参数 md_filename=report.md。
不要再次调用 generate_markdown，不要调用 task、搜索、数据库或知识库工具。
"""


def _blocked(*_args, **_kwargs):
    raise RuntimeError("smoke 禁止联网、业务库和 Embedding")


async def _drain(agent, payload) -> list[dict]:
    pending: list[dict] = []
    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": RECURSION_LIMIT,
    }
    async for chunk in agent.astream(payload, config=config):
        interrupts = chunk.get("__interrupt__")
        if not interrupts:
            continue
        for item in interrupts:
            value = getattr(item, "value", None) or {}
            for action in value.get("action_requests", []):
                pending.append(
                    {
                        "tool": action.get("name"),
                        "description": action.get("description", ""),
                    }
                )
    return pending


async def main() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.llm_api_key:
        raise SystemExit("missing llm key")

    output_root = project_root_path / "output"
    session_dir = session_directory(output_root, THREAD_ID)
    session_dir.mkdir(parents=True, exist_ok=True)
    session_token = set_session_context(str(session_dir))
    thread_token = set_thread_context(THREAD_ID)
    budget = ContextWindowMiddleware(
        max_chars=8000,
        max_model_calls=MAX_MODEL_CALLS,
        stop_after="Markdown文件",
    )

    import app.tools.tavily_tool as tavily_tool
    import app.tools.db_tools as db_tools
    import app.rag.embedding_client as embedding_client

    tavily_tool.tavily_search = _blocked
    db_tools.execute_sql_query = _blocked
    embedding_client.embed_texts = _blocked

    try:
        await init_checkpointer()
        agent = create_deep_agent(
            model=model,
            system_prompt="你是本地验收助手。只使用用户给出的事实。generate_markdown 只能调用一次；成功后下一条必须是 convert_md_to_pdf。",
            tools=[generate_markdown, convert_md_to_pdf],
            middleware=[budget],
            checkpointer=await init_checkpointer(),
            interrupt_on={"convert_md_to_pdf": True},
            subagents=[],
        )
        try:
            pending = await asyncio.wait_for(
                _drain(agent, {"messages": [{"role": "user", "content": FACTS}]}),
                timeout=TIMEOUT_SECONDS,
            )
        except RuntimeError as exc:
            if "停止继续调用模型" not in str(exc) and "模型调用超过上限" not in str(exc):
                raise
            pending = []
        if not any(item["tool"] == "convert_md_to_pdf" for item in pending):
            budget.stop_after = None
            budget._stop_tool_seen = False
            budget.max_model_calls = budget.model_calls + 1
            pending = await asyncio.wait_for(
                _drain(
                    agent,
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "只调用 convert_md_to_pdf，md_filename=report.md。不要再写 Markdown。",
                            }
                        ]
                    },
                ),
                timeout=40,
            )
        print("approval", [item["tool"] for item in pending])
        budget.stop_after = None
        budget._stop_tool_seen = False
        budget.max_model_calls = budget.model_calls + 2
        await asyncio.wait_for(
            _drain(agent, Command(resume={"decisions": [{"type": "approve"}]})),
            timeout=TIMEOUT_SECONDS,
        )
        md_files = list(session_dir.rglob("*.md"))
        pdf_files = list(session_dir.rglob("*.pdf"))
        print("markdown", len(md_files), "pdf", len(pdf_files), "model_calls", budget.model_calls)
        if not md_files or not pdf_files:
            raise RuntimeError("未生成完整产物")
        marker = "本地验收样例，不是调研结论"
        text = md_files[0].read_text(encoding="utf-8")
        if marker not in text:
            md_files[0].write_text(f"> {marker}\n\n{text}", encoding="utf-8")
            from app.utils.word_converter import convert_md_to_pdf as rebuild_pdf

            rebuild_pdf(md_files[0], pdf_files[0])
        text = md_files[0].read_text(encoding="utf-8")
        if marker not in text:
            raise RuntimeError("产物缺少本地验收标记")
        print("sample_marked", True)
        print("session", THREAD_ID)
    finally:
        reset_session_context(session_token, thread_token)
        await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
