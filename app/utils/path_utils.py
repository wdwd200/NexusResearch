"""
文件路径解析工具

负责把模型或工具返回的虚拟路径、上传文件路径和相对路径统一转换为本地绝对路径。
后续文件读取、Markdown 生成和 PDF 转换工具都可以复用这里的解析规则。

约束是物理的：给出 session_dir 时，解析结果必须落在该目录内。
../、盘符绝对路径、会话外绝对路径都会被拒绝，不依赖提示词。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# 会话 ID 只允许 UUID 或短安全标识，避免被拼进目录名后穿越。
THREAD_ID_PATTERN = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[A-Za-z0-9][A-Za-z0-9_-]{0,63})$"
)

_VIRTUAL_PREFIXES = ("/workspace", "/mnt/data", "/home/user")


class PathEscapeError(ValueError):
    """请求路径逃出了允许的会话目录。"""


def is_valid_thread_id(thread_id: str | None) -> bool:
    """判断会话 ID 能否安全用作目录名和路由参数。"""
    if not thread_id or not isinstance(thread_id, str):
        return False
    if thread_id in {".", ".."}:
        return False
    if "/" in thread_id or "\\" in thread_id or "\x00" in thread_id:
        return False
    return bool(THREAD_ID_PATTERN.fullmatch(thread_id))


def session_directory(output_root: Path, thread_id: str) -> Path:
    """返回 output/session_{thread_id}，并确认它仍在 output 根目录下。"""
    if not is_valid_thread_id(thread_id):
        raise PathEscapeError("非法会话 ID")
    root = output_root.resolve()
    session = (root / f"session_{thread_id}").resolve()
    if not session.is_relative_to(root):
        raise PathEscapeError("会话目录越界")
    return session


def _normalize_logical(filename: str) -> str:
    text = str(filename or "").strip().replace("\\", "/")
    for prefix in _VIRTUAL_PREFIXES:
        if text == prefix or text.startswith(prefix + "/"):
            text = text[len(prefix) :].lstrip("/")
            break
    # 上传暂存目录只取文件名，真实文件已复制进当前会话目录
    if "updated/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


def _contains_parent_segment(text: str) -> bool:
    parts = [part for part in text.replace("\\", "/").split("/") if part not in {"", "."}]
    return ".." in parts


def resolve_path(filename: str, session_dir: str | None = None) -> str:
    """
    解析文件路径，并强制把任务产物限制在当前会话目录中。

    :param filename: 模型、工具或用户传入的文件名/路径
    :param session_dir: 当前任务的会话目录；给出时结果必须位于其内
    :return: 解析后的绝对路径
    :raises PathEscapeError: 路径包含 ..、落在会话外，或未提供会话目录却给出绝对路径
    """
    if filename is None or not str(filename).strip():
        raise PathEscapeError("路径为空")

    raw = str(filename)
    if "\x00" in raw or _contains_parent_segment(raw):
        raise PathEscapeError("路径包含非法片段")

    logical = _normalize_logical(raw)
    if not logical or _contains_parent_segment(logical):
        raise PathEscapeError("路径包含非法片段")

    path = Path(logical)
    path_str = logical
    is_unix_abs = path_str.startswith("/")
    looks_absolute = path.is_absolute() or (os.name == "nt" and is_unix_abs and bool(path.drive))

    if not session_dir:
        # 没有会话目录时只解析当前工作目录内的相对路径，拒绝任意绝对路径
        if looks_absolute or is_unix_abs:
            raise PathEscapeError("缺少会话目录，拒绝绝对路径")
        return str(path.resolve())

    session_path = Path(session_dir).resolve()
    session_name = session_path.name

    if looks_absolute or is_unix_abs:
        # Windows 上 "/etc/passwd" 没有盘符，但不能把它改写成会话内相对路径
        if os.name == "nt" and is_unix_abs and not path.drive:
            raise PathEscapeError("拒绝无盘符绝对路径")
        else:
            candidate = Path(path_str)
            try:
                relative = candidate.resolve().relative_to(session_path)
            except ValueError as exc:
                raise PathEscapeError("拒绝会话目录之外的绝对路径") from exc
            candidate = session_path / relative
    else:
        parts = [part for part in path.parts if part not in {"", "."}]
        # 模型常把 session 名或 output 前缀再拼一次，只保留文件名，避免嵌套出目录
        if session_name in parts or (parts and parts[0] == "output"):
            candidate = session_path / path.name
        else:
            candidate = session_path / path

    resolved = candidate.resolve()
    if resolved != session_path and not resolved.is_relative_to(session_path):
        raise PathEscapeError("路径越出会话目录")
    return _fix_nested_session_path(resolved, session_path, session_name)


def resolve_within(root: Path, relative: str) -> Path:
    """把相对路径解析到 root 内，用于下载/列目录。"""
    if not relative or not str(relative).strip():
        raise PathEscapeError("路径为空")
    if _contains_parent_segment(str(relative)) or "\x00" in str(relative):
        raise PathEscapeError("路径包含非法片段")
    base = root.resolve()
    candidate = (base / relative).resolve()
    if candidate != base and not candidate.is_relative_to(base):
        raise PathEscapeError("路径越出允许目录")
    return candidate


def _fix_nested_session_path(
    full_path: Path,
    session_path: Path,
    session_name: str,
) -> str:
    """修正 session_xxx/session_xxx/file.md 这类重复嵌套路径。"""
    parts = full_path.parts
    for index in range(len(parts) - 1):
        if parts[index] == session_name and parts[index + 1] == session_name:
            flattened = session_path / full_path.name
            if flattened != session_path and not flattened.is_relative_to(session_path):
                raise PathEscapeError("路径越出会话目录")
            return str(flattened)
    return str(full_path)
