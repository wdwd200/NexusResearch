"""
主智能体组装与异步执行模块

把模型、主提示词、文件类工具和三个专家子智能体组装成 DeepAgent，
并提供启动任务与断点恢复两个入口。

核心设计：
1. checkpointer 采用 SQLite 持久化状态账本，
   服务重启后仍能按 thread_id 恢复；
2. 在模型调用前挂载 ContextWindowMiddleware，做滑动窗口裁剪和工具消息配对修复；
3. 对正式 PDF 生成启用人工确认，命中后推送大纲预览，等待前端 resume。
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.agent.checkpointer import get_checkpointer, init_checkpointer
from app.agent.context_manager import ContextWindowMiddleware
from app.agent.llm import model
from app.agent.prompts import main_agent_content
from app.agent.subagents.database_query_agent import database_query_agent
from app.agent.subagents.knowledge_base_agent import knowledge_base_agent
from app.agent.subagents.network_search_agent import network_search_agent
from app.api.context import (
    reset_session_context,
    set_session_context,
    set_thread_context,
)
from app.api.monitor import monitor
from app.config import get_settings
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.tools.upload_file_read_tool import read_file_content
from app.utils.path_utils import is_valid_thread_id, session_directory

logger = logging.getLogger(__name__)

settings = get_settings()

# 项目根目录：当前文件位于 app/agent/main_agent.py，parents[2] 才是仓库根目录。
# 产物目录必须落在仓库根的 output/updated 下，否则容器卷挂载会错位到 app/ 内部。
project_root_path = Path(__file__).parents[2].resolve()

_agent: Any = None
_agent_checkpointer: Any = None
_agent_lock = asyncio.Lock()


def _build_agent() -> Any:
    """用已初始化的状态账本组装主智能体。"""
    interrupt_on = {
        tool_name: True for tool_name in settings.hitl_interrupt_tools
    }
    return create_deep_agent(
        model=model,
        system_prompt=main_agent_content["system_prompt"],
        tools=[generate_markdown, convert_md_to_pdf, read_file_content],
        middleware=[
            # 上下文窗口治理 + 模型调用步数上限：中转站慢，必须给任务一个有限预算，
            # 否则模型会反复重写文件直到超时。达到上限时抛出明确错误而不是静默卡住。
            # 注意不能设置 stop_after=generate_markdown：那会在 Markdown 完成后
            # 立刻终止，导致 PDF 转换和人工确认永远无法触发。
            ContextWindowMiddleware(
                max_chars=settings.context_max_chars,
                max_model_calls=settings.max_model_calls,
            )
        ],
        checkpointer=get_checkpointer(),
        interrupt_on=interrupt_on,
        subagents=[database_query_agent, network_search_agent, knowledge_base_agent],
    )


async def get_main_agent() -> Any:
    """
    懒加载主智能体；同一进程内只构建一次。

    如果状态账本被关闭后重新初始化（例如测试环境的 lifespan 重入），
    缓存的 Agent 仍然持有旧连接，必须检测到不一致后重建。
    """
    global _agent, _agent_checkpointer
    async with _agent_lock:
        saver = await init_checkpointer()
        if _agent is None or _agent_checkpointer is not saver:
            _agent = _build_agent()
            _agent_checkpointer = saver
        return _agent


def _prepare_session_dirs(session_id: str) -> tuple[Path, str, str, str]:
    """
    准备会话工作目录并复制上传附件。

    :return: (session_dir, 绝对路径, 相对路径, 上传文件提示)
    """
    if not is_valid_thread_id(session_id):
        raise ValueError("非法会话 ID")
    session_dir = session_directory(project_root_path / "output", session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    session_dir_str = str(session_dir).replace("\\", "/")
    relative_session_dir_str = str(session_dir.relative_to(project_root_path)).replace(
        "\\", "/"
    )

    updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
    updated_info_prompt = ""
    if updated_dir_path.exists():
        files = sorted(f.name for f in updated_dir_path.iterdir() if f.is_file())
        if files:
            for filename in files:
                shutil.copy2(updated_dir_path / filename, session_dir / filename)
            updated_info_prompt = (
                "\n    [已上传文件] 已加载到工作目录:\n"
                + "\n".join([f"    - {f}" for f in files])
                + "\n    请优先使用工具（read_file_content）读取并参考这些文件。"
            )

    return session_dir, session_dir_str, relative_session_dir_str, updated_info_prompt


def _build_path_instruction(relative_session_dir_str: str, updated_info_prompt: str) -> str:
    """运行时注入工作目录契约，约束模型只在当前会话目录读写。"""
    return f"""
    【工作环境指令】
    工作目录: {relative_session_dir_str}
    {updated_info_prompt}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """


def _handle_chunk(node_name: str, state: dict[str, Any]) -> None:
    """把模型节点产生子智能体调用或最终文本上报给前端。"""
    messages = state.get("messages") or []
    if not messages or not isinstance(messages, list):
        return

    last_msg = messages[-1]
    if node_name != "model" or not isinstance(last_msg, AIMessage):
        return

    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.get("name") == "task":
                args = tool_call.get("args") or {}
                monitor.report_assistant(
                    args.get("subagent_type", "unknown"),
                    {"description": args.get("description", "")},
                )
        return

    content = last_msg.content
    if content:
        text = content if isinstance(content, str) else str(content)
        logger.info("主智能体输出结果片段：%s", text[:100])
        monitor.report_task_result(text)


def _extract_interrupt(chunk: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    从 astream 片段中提取人工确认请求。

    HumanInTheLoopMiddleware 中断时会出现 __interrupt__ 键，其 value 包含
    action_requests 与 review_configs。
    """
    interrupts = chunk.get("__interrupt__")
    if not interrupts:
        return None

    requests: list[dict[str, Any]] = []
    for item in interrupts:
        value = getattr(item, "value", None) or {}
        for action in value.get("action_requests", []):
            requests.append(
                {
                    "tool": action.get("name"),
                    "args": action.get("args") or {},
                    "description": action.get("description", ""),
                }
            )
    return requests


async def _report_state_failure(agent: Any, config: dict[str, Any], stage: str) -> bool:
    """
    检查运行结束后的图状态，把被吞掉的节点异常显式报给前端。

    LangGraph 在子图/工具节点抛错时不会让 astream 直接抛出，而是把错误挂在
    task.error 上，图停在某个节点。若不检查，前端只会看到任务“没有下文”。
    """
    try:
        state = await agent.aget_state(config)
    except Exception as exc:  # 状态读取失败不覆盖原始错误
        logger.warning("读取图状态失败：%s", exc)
        return False

    for task in state.tasks or []:
        error = getattr(task, "error", None)
        if error:
            message = f"{stage}失败：{error}"
            logger.error(message)
            monitor._emit("error", message)
            return True
    return False


def _build_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


async def run_deep_agent(task_query: str, session_id: str) -> None:
    """
    启动一次新的投研任务。

    :param task_query: 用户原始任务
    :param session_id: 会话 ID，同时作为 thread_id、输出目录和 WebSocket 频道
    """
    logger.info("开始执行会话，session_id=%s", session_id)

    _session_dir, session_dir_str, relative_dir, updated_prompt = _prepare_session_dirs(
        session_id
    )
    path_instruction = _build_path_instruction(relative_dir, updated_prompt)

    session_dir_token = set_session_context(session_dir_str)
    thread_id_token = set_thread_context(session_id)
    monitor.report_session_dir(session_dir_str)

    agent = await get_main_agent()
    config = _build_config(session_id)

    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": task_query + path_instruction}]},
            config=config,
        ):
            pending = _extract_interrupt(chunk)
            if pending:
                # 中断点已持久化到状态账本，前端确认后可用同一 thread_id 恢复
                monitor.report_awaiting_approval(pending)
                continue

            for node_name, state in chunk.items():
                if isinstance(state, dict):
                    _handle_chunk(node_name, state)

        # astream 正常结束不代表成功：子图异常会挂在 task.error 上
        await _report_state_failure(agent, config, "任务执行")

    except asyncio.CancelledError:
        monitor.report_task_cancelled()
        raise
    except Exception as exc:
        logger.exception("主智能体执行异常")
        monitor._emit("error", f"执行主智能发生异常信息：{exc}")
    finally:
        reset_session_context(session_dir_token, thread_id_token)


async def resume_deep_agent(
    thread_id: str,
    decisions: list[dict[str, Any]],
    feedback: str = "",
) -> None:
    """
    从人工确认断点恢复执行。

    :param thread_id: 与首次任务相同的 thread_id
    :param decisions: HumanInTheLoop 决策列表，形如 [{"type": "approve"}]
    :param feedback: 用户修改意见；非空时作为附加指令注入恢复请求
    """
    logger.info("恢复会话，thread_id=%s", thread_id)

    if not is_valid_thread_id(thread_id):
        raise ValueError("非法会话 ID")
    session_dir = session_directory(project_root_path / "output", thread_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    session_dir_str = str(session_dir).replace("\\", "/")

    session_dir_token = set_session_context(session_dir_str)
    thread_id_token = set_thread_context(thread_id)

    agent = await get_main_agent()
    config = _build_config(thread_id)

    effective_decisions = decisions
    if feedback.strip():
        # HumanInTheLoop 的 reject 决策支持携带 message，模型会收到这条反馈并重新生成，
        # 比直接改写状态消息更安全，因为中断恢复依赖原始检查点结构。
        effective_decisions = [
            {
                "type": "reject",
                "message": f"用户未通过当前方案，修改意见：{feedback.strip()}",
            }
            for _ in decisions
        ]
        monitor.report_approval_accepted(thread_id, feedback)

    # 恢复请求必须用同一个 thread_id 才能命中挂起检查点
    payload: Any = Command(resume={"decisions": effective_decisions})

    try:
        async for chunk in agent.astream(payload, config=config):
            pending = _extract_interrupt(chunk)
            if pending:
                monitor.report_awaiting_approval(pending)
                continue

            for node_name, state in chunk.items():
                if isinstance(state, dict):
                    _handle_chunk(node_name, state)

        await _report_state_failure(agent, config, "恢复执行")

    except asyncio.CancelledError:
        monitor.report_task_cancelled()
        raise
    except Exception as exc:
        logger.exception("恢复执行异常")
        monitor._emit("error", f"恢复执行发生异常：{exc}")
    finally:
        reset_session_context(session_dir_token, thread_id_token)


async def get_pending_approval(thread_id: str) -> list[dict[str, Any]] | None:
    """
    查询某个 thread 是否处于等待人工确认状态。

    :return: 待确认动作列表；没有挂起任务时返回 None
    """
    agent = await get_main_agent()
    state = await agent.aget_state(_build_config(thread_id))
    if not state.next:
        return None

    for task in state.tasks or []:
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None) or {}
            requests = [
                {
                    "tool": action.get("name"),
                    "args": action.get("args") or {},
                    "description": action.get("description", ""),
                }
                for action in value.get("action_requests", [])
            ]
            if requests:
                return requests
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_deep_agent("从网络查询机器人信息，并生成Markdown文件", "test_session_001"))
