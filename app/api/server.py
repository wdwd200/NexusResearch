"""
FastAPI 接口层与项目闭环入口

负责承接前端的任务提交、任务取消、断点恢复、文件上传/下载和 WebSocket 长连接。
HTTP 接口只做轻量调度，真正的 Agent 执行放到后台任务中；执行进度、工具调用、
待审批动作和最终结果由 monitor 按 thread_id 推送给前端。

核心职责：
1. lifespan 中初始化并释放 SQLite 状态账本，支撑跨重启的断点恢复；
2. 提供 /api/task/{thread_id}/resume 与 /api/task/{thread_id}/status；
3. 统一 Bearer Token 鉴权依赖，保护公网部署时的模型额度。
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.main_agent import (
    get_pending_approval,
    project_root_path,
    resume_deep_agent,
    run_deep_agent,
)
from app.api.monitor import manager
from app.config import get_settings
from app.utils.path_utils import PathEscapeError, is_valid_thread_id, resolve_within

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    服务生命周期入口。

    启动时初始化状态账本、绑定 WebSocket 事件循环；退出时先取消所有后台任务，
    再释放数据库连接，避免任务在连接关闭后继续读写检查点。
    """
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)
    await init_checkpointer()
    logger.info("服务启动完成，WebSocket loop=%s", id(loop))
    try:
        yield
    finally:
        for task in list(active_tasks.values()):
            if not task.done():
                task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks.values(), return_exceptions=True)
            active_tasks.clear()
        await close_checkpointer()
        logger.info("服务已关闭，状态账本连接已释放")


app = FastAPI(
    title="NexusResearch API",
    description="多智能体协同投研与自动化报告生成系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 保存 thread_id -> 后台 Agent 任务，用于同一会话任务替换和主动取消
active_tasks: dict[str, asyncio.Task] = {}

# output 保存每个会话最终工作区，前端只允许从这里浏览和下载生成文件
output_dir = project_root_path / "output"
output_dir.mkdir(exist_ok=True)

# updated 暂存用户上传文件，run_deep_agent 启动时会复制到对应 output/session_xxx
updated_dir = project_root_path / "updated"
updated_dir.mkdir(exist_ok=True)

# 生产部署时前端与 API 同域，仍然放开跨域以便独立部署的前端调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def access_token_matches(token: str | None) -> bool:
    """
    单进程演示口令校验。

    未配置 ACCESS_TOKEN 时放行。配置后只是同一把演示口令，不是用户隔离。
    """
    expected = settings.access_token
    if not expected:
        return True
    return bool(token) and token == expected


def require_access_token(
    authorization: str | None = Header(default=None),
    x_access_token: str | None = Header(default=None, alias="X-Access-Token"),
) -> None:
    """
    HTTP 鉴权依赖。

    ACCESS_TOKEN 未配置时直接放行（本地开发）；配置后要求请求头携带
    `Authorization: Bearer <token>` 或 `X-Access-Token: <token>`。
    下载链接不走这个依赖，改为短时票据，避免把口令写进 URL。
    """
    expected = settings.access_token
    if not expected:
        return

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_access_token:
        token = x_access_token.strip()

    if not access_token_matches(token):
        raise HTTPException(status_code=401, detail="访问令牌无效或缺失")


def _ticket_secret() -> bytes:
    """票据签名材料。未配置口令时使用进程内随机值，重启后旧票据失效。"""
    if settings.access_token:
        return settings.access_token.encode("utf-8")
    return _ephemeral_ticket_secret


_ephemeral_ticket_secret = uuid.uuid4().hex.encode("utf-8")
DOWNLOAD_TICKET_TTL_SECONDS = 300


def issue_download_ticket(thread_id: str, relative_path: str, now: float | None = None) -> str:
    """签发绑定会话和相对路径的短时下载票据，不暴露 ACCESS_TOKEN。"""
    issued = int(now if now is not None else time.time())
    payload = f"{thread_id}|{relative_path}|{issued}"
    digest = hmac.new(_ticket_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{issued}.{digest}"


def verify_download_ticket(
    thread_id: str,
    relative_path: str,
    ticket: str | None,
    now: float | None = None,
) -> bool:
    """校验下载票据。未配置 ACCESS_TOKEN 的本地演示也要求票据，防止裸链被转发后长期有效。"""
    if not ticket or "." not in ticket:
        return False
    issued_text, digest = ticket.split(".", 1)
    try:
        issued = int(issued_text)
    except ValueError:
        return False
    current = int(now if now is not None else time.time())
    if issued > current + 30 or current - issued > DOWNLOAD_TICKET_TTL_SECONDS:
        return False
    expected = issue_download_ticket(thread_id, relative_path, now=issued)
    return hmac.compare_digest(expected, f"{issued}.{digest}")


def require_thread_id(thread_id: str) -> str:
    """拒绝可穿越目录或污染路由的会话 ID。"""
    if not is_valid_thread_id(thread_id):
        raise HTTPException(status_code=400, detail="非法会话 ID")
    return thread_id


class TaskRequest(BaseModel):
    """前端启动任务时提交的请求体。"""

    query: str = Field(min_length=1, description="研搜任务描述")
    thread_id: str | None = Field(default=None, description="会话 ID，可复用")

    @field_validator("thread_id")
    @classmethod
    def _check_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not is_valid_thread_id(value):
            raise ValueError("非法会话 ID")
        return value


class ResumeRequest(BaseModel):
    """人工确认后恢复执行的请求体。"""

    decision: str = Field(
        default="approve",
        description="approve 继续执行；reject 拒绝并按 feedback 重新生成",
    )
    feedback: str = Field(default="", description="拒绝时给出的修改意见")


class TaskStatusResponse(BaseModel):
    """任务状态查询结果。"""

    thread_id: str
    running: bool
    awaiting_approval: bool
    actions: list[dict] = Field(default_factory=list)
    persistence: str = "approval_checkpoint_only"


def _forget_task(thread_id: str, task: asyncio.Task) -> None:
    """
    清理已结束任务的登记关系。

    done_callback 触发时，active_tasks 中可能已经被新任务替换；只有仍是同一个
    task 时才删除，避免误清理同 thread_id 下刚启动的新任务。
    """
    if active_tasks.get(thread_id) is task:
        active_tasks.pop(thread_id, None)


def _spawn_task(thread_id: str, coro) -> asyncio.Task:
    """启动后台任务并登记，自动清理引用。"""
    old_task = active_tasks.get(thread_id)
    if old_task and not old_task.done():
        old_task.cancel()

    task = asyncio.create_task(coro)
    active_tasks[thread_id] = task
    task.add_done_callback(lambda finished: _forget_task(thread_id, finished))
    return task


@app.post("/api/task", dependencies=[Depends(require_access_token)])
async def run_task(request: TaskRequest):
    """
    启动一次投研任务。

    HTTP 请求只负责创建后台协程并立即返回，后续执行轨迹、子智能体调用和最终
    答案都会由 monitor 通过 `/ws/{thread_id}` 推送给同一会话的前端。
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    require_thread_id(thread_id)
    _spawn_task(thread_id, run_deep_agent(request.query, thread_id))
    return {"status": "started", "thread_id": thread_id}


@app.post("/api/task/{thread_id}/resume", dependencies=[Depends(require_access_token)])
async def resume_task(thread_id: str, request: ResumeRequest):
    """
    人工确认后从断点恢复执行。

    前端在大纲预览界面点击“确认生成”或提交修改意见后调用本接口，
    必须携带与首次任务相同的 thread_id，才能命中状态账本中的挂起检查点。
    """
    thread_id = require_thread_id(thread_id)
    actions = await get_pending_approval(thread_id)
    if not actions:
        raise HTTPException(status_code=409, detail="该会话没有等待确认的任务")

    # 已有恢复任务在跑时也拒绝重复提交，避免同一检查点被并发恢复
    running = active_tasks.get(thread_id)
    if running and not running.done():
        raise HTTPException(status_code=409, detail="该会话正在执行中，请勿重复确认")

    if request.decision not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="decision 只支持 approve 或 reject")

    decisions = [{"type": request.decision}]
    _spawn_task(
        thread_id,
        resume_deep_agent(
            thread_id,
            decisions=decisions,
            feedback=request.feedback if request.decision == "reject" else "",
        ),
    )
    return {"status": "resumed", "thread_id": thread_id, "decision": request.decision}


@app.get(
    "/api/task/{thread_id}/status",
    response_model=TaskStatusResponse,
    dependencies=[Depends(require_access_token)],
)
async def task_status(thread_id: str):
    """
    查询会话状态。

    前端刷新页面或重连 WebSocket 后，用本接口判断任务是否仍在执行、
    是否停在人工确认界面。SQLite 只恢复待审批检查点，不承诺把已中断的
    模型调用重新放进通用任务队列。
    """
    thread_id = require_thread_id(thread_id)
    task = active_tasks.get(thread_id)
    try:
        actions = await get_pending_approval(thread_id)
    except Exception as exc:
        # 状态查询是前端刷新时的探活请求，读检查点失败不应让整个页面报 500；
        # 返回 awaiting_approval=false 并记录日志，由前端继续轮询重试。
        logger.warning("查询待审批状态失败 thread_id=%s: %s", thread_id, exc)
        actions = None
    return TaskStatusResponse(
        thread_id=thread_id,
        running=bool(task and not task.done()),
        awaiting_approval=bool(actions),
        actions=actions or [],
        persistence="approval_checkpoint_only",
    )


@app.post("/api/task/{thread_id}/cancel", dependencies=[Depends(require_access_token)])
async def cancel_task(thread_id: str):
    thread_id = require_thread_id(thread_id)
    """
    取消指定 thread_id 对应的后台任务。

    注意：取消会向 asyncio.Task 注入 CancelledError。若底层第三方工具正在执行不可中断
    的同步阻塞调用，任务可能需要等该调用返回后才会真正结束。
    """
    task = active_tasks.get(thread_id)
    if not task or task.done():
        active_tasks.pop(thread_id, None)
        raise HTTPException(status_code=404, detail="任务不存在或已结束")

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.CancelledError:
        _forget_task(thread_id, task)
        return {"status": "cancelled", "thread_id": thread_id}
    except TimeoutError:
        return {"status": "cancelling", "thread_id": thread_id}
    except Exception as exc:
        _forget_task(thread_id, task)
        return {"status": "cancelled", "thread_id": thread_id, "message": str(exc)}

    _forget_task(thread_id, task)
    return {"status": "cancelled", "thread_id": thread_id}


@app.post("/api/upload", dependencies=[Depends(require_access_token)])
async def upload_files(files: list[UploadFile] = File(...), thread_id: str = Form(...)):
    """
    接收用户上传的一个或多个文件，保存到 `updated/session_{thread_id}`。

    上传文件按会话隔离，Agent 在后续任务中会把它们复制到本次工作目录。
    """
    thread_id = require_thread_id(thread_id)
    target_dir = resolve_within(updated_dir, f"session_{thread_id}")
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        # 只保留文件名，避免客户端传入带路径的名字造成目录穿越
        safe_name = Path(file.filename or "upload.bin").name
        file_path = target_dir / safe_name
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(safe_name)

    return {"status": "uploaded", "files": saved_files}


def _session_output_dir(thread_id: str) -> Path:
    """定位某个会话的产物目录，拒绝逃出 output。"""
    require_thread_id(thread_id)
    try:
        return resolve_within(output_dir, f"session_{thread_id}")
    except PathEscapeError as exc:
        raise HTTPException(status_code=400, detail="非法会话目录") from exc


def _relative_artifact(thread_id: str, relative_path: str) -> Path:
    session_dir = _session_output_dir(thread_id)
    try:
        return resolve_within(session_dir, relative_path)
    except PathEscapeError as exc:
        raise HTTPException(status_code=400, detail="拒绝访问: 只能访问当前会话产物") from exc


@app.get("/api/download")
async def download_file(thread_id: str, path: str, ticket: str | None = None):
    """
    下载当前会话产物。

    只接受 thread_id + 会话内相对路径。`<a href>` 不能带请求头，
    因此使用绑定路径的短时票据，而不是把 ACCESS_TOKEN 放进查询参数。
    """
    require_thread_id(thread_id)
    abs_path = _relative_artifact(thread_id, path)
    if not verify_download_ticket(thread_id, path, ticket):
        raise HTTPException(status_code=401, detail="下载票据无效或已过期")
    if not abs_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(abs_path, filename=abs_path.name)


@app.get("/api/files", dependencies=[Depends(require_access_token)])
async def list_files(thread_id: str):
    """
    列出指定会话目录下的生成文件。

    返回会话内相对路径，前端再换取短时下载票据，不再暴露宿主机绝对路径。
    """
    abs_path = _session_output_dir(thread_id)
    if not abs_path.exists():
        return {"files": [], "thread_id": thread_id, "session_ready": False}

    files = []
    try:
        for file_path in abs_path.rglob("*"):
            if not file_path.is_file():
                continue
            stat = file_path.stat()
            relative = file_path.relative_to(abs_path).as_posix()
            files.append(
                {
                    "name": file_path.name,
                    "type": "file",
                    "path": relative,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "ticket": issue_download_ticket(thread_id, relative),
                }
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法列出会话产物") from exc

    files.sort(key=lambda item: item.get("mtime", 0), reverse=True)
    return {"files": files, "thread_id": thread_id, "session_ready": True}


@app.get("/api/health")
async def health():
    """健康检查：部署后用于确认服务、状态账本与模型并发上限。"""
    return {
        "status": "ok",
        "model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "llm_max_concurrency": settings.llm_max_concurrency,
        "checkpoint_db": str(settings.checkpoint_db_path),
        "active_tasks": len(active_tasks),
    }


@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """
    WebSocket 实时通讯接口。

    连接建立后按 thread_id 注册，monitor 的事件会定向推送到对应页面。
    配置了 ACCESS_TOKEN 时，必须在首条文本消息里提交同一把演示口令。
    这不是用户隔离，只阻止未持有演示口令的连接占用推送通道。
    """
    if not is_valid_thread_id(thread_id):
        await websocket.close(code=1008, reason="非法会话 ID")
        return

    # 浏览器 WebSocket 不能自定义请求头。演示口令只放在握手查询参数里做受控校验，
    # 不写入下载链接，也不当作多用户隔离。
    if settings.access_token and not access_token_matches(websocket.query_params.get("token")):
        await websocket.close(code=1008, reason="访问令牌无效")
        return

    await manager.connect(websocket, thread_id)

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json(
                {"type": "pong", "message": f"服务端已收到: {data}"}
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket, thread_id)
    except Exception as exc:
        logger.warning("WebSocket 连接异常 %s: %s", thread_id, exc)
        manager.disconnect(websocket, thread_id)


if __name__ == "__main__":
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8002, reload=True)
