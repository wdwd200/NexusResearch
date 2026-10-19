"""
状态账本（Checkpointer）管理模块

用 langgraph-checkpoint-sqlite 把每个 thread_id 的执行状态持久化到 SQLite。
这里只支撑人工确认挂起后的断点恢复，不是通用任务队列：进程内正在跑的
模型调用不会在重启后自动重放。

注意两点实现约束：
1. 必须使用 AsyncSqliteSaver：同步版 SqliteSaver 不支持 astream，会直接抛
   NotImplementedError；
2. 连接是长生命周期的，由 FastAPI lifespan 负责启动时 setup、关闭时释放。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings

logger = logging.getLogger(__name__)

_saver: AsyncSqliteSaver | None = None
_connection: aiosqlite.Connection | None = None
_lock = asyncio.Lock()


async def init_checkpointer() -> AsyncSqliteSaver:
    """
    初始化全局状态账本，幂等。

    :return: 可直接传给 create_deep_agent 的 AsyncSqliteSaver
    """
    global _saver, _connection

    async with _lock:
        if _saver is not None:
            return _saver

        db_path = get_settings().checkpoint_db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False 允许同一连接在事件循环的不同任务间复用
        _connection = await aiosqlite.connect(str(db_path), check_same_thread=False)
        try:
            # WAL 让并发读写不会互相阻塞，这里顺带设置较宽松的同步级别
            await _connection.execute("PRAGMA journal_mode=WAL")
            await _connection.execute("PRAGMA synchronous=NORMAL")
            await _connection.commit()
        except Exception as exc:  # pragma: no cover - 个别文件系统不支持 WAL
            logger.warning("启用 WAL 失败，继续使用默认日志模式：%s", exc)

        _saver = AsyncSqliteSaver(_connection)
        await _saver.setup()
        logger.info("状态账本已就绪：%s", db_path)
        return _saver


async def close_checkpointer() -> None:
    """关闭状态账本连接，供 FastAPI lifespan 退出时调用。"""
    global _saver, _connection

    async with _lock:
        if _connection is not None:
            await _connection.close()
        _connection = None
        _saver = None
        logger.info("状态账本连接已关闭")


def get_checkpointer() -> AsyncSqliteSaver:
    """获取已初始化的状态账本；未初始化时给出明确错误。"""
    if _saver is None:
        raise RuntimeError("状态账本尚未初始化，请先调用 init_checkpointer()")
    return _saver
