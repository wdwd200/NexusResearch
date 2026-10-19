"""无密钥回归：路径、鉴权、上下文配对、只读 SQL。不连接真实数据库，不调用模型。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("RERANK_PROVIDER", "local")
os.environ.setdefault(
    "CHECKPOINT_DB_PATH",
    str(Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".") / "nexus-test-checkpoints.sqlite"),
)

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.context_manager import ContextWindowMiddleware, build_context_window, pair_tool_messages
from app.api import server
from app.config import get_settings
from app.tools.db_tools import _assert_readonly
from app.utils.path_utils import PathEscapeError, is_valid_thread_id, resolve_path


class PathConstraintTests(unittest.TestCase):
    def test_rejects_parent_and_outside_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session_demo"
            session.mkdir()
            (session / "note.md").write_text("ok", encoding="utf-8")

            inside = resolve_path("note.md", str(session))
            self.assertEqual(Path(inside), (session / "note.md").resolve())
            nested = resolve_path("sub/report.md", str(session))
            self.assertTrue(Path(nested).is_relative_to(session.resolve()))

            for unsafe in ("../secret.txt", "..\\secret.txt", "/etc/passwd", "C:/Windows/win.ini"):
                with self.assertRaises(PathEscapeError):
                    resolve_path(unsafe, str(session))

    def test_thread_id_is_directory_safe(self) -> None:
        self.assertTrue(is_valid_thread_id("550e8400-e29b-41d4-a716-446655440000"))
        self.assertTrue(is_valid_thread_id("manual-1"))
        for unsafe in ("../x", "a/b", "a\\b", "", ".", "..", "has space"):
            self.assertFalse(is_valid_thread_id(unsafe))


class ContextWindowTests(unittest.TestCase):
    def test_keeps_complete_tool_group_inside_budget(self) -> None:
        huge = "x" * 9000
        messages = [
            SystemMessage(content="old-system"),
            HumanMessage(content="first-user-" + ("q" * 3000)),
            AIMessage(
                content="",
                tool_calls=[{"name": "read", "args": {"q": "a"}, "id": "call-1", "type": "tool_call"}],
            ),
            ToolMessage(content=huge, tool_call_id="call-1"),
            HumanMessage(content="latest-user"),
        ]
        window = build_context_window(messages, "latest-system", max_chars=5000)
        types = [item.type for item in window]
        self.assertEqual(types[0], "system")
        self.assertEqual(window[0].content, "latest-system")
        self.assertEqual(types.count("system"), 1)
        tool_index = types.index("tool")
        self.assertEqual(types[tool_index - 1], "ai")
        self.assertLess(len(window[tool_index].content), len(huge))
        self.assertIn("latest-user", [item.content for item in window if item.type == "human"])
        self.assertNotIn("first-user", "".join(item.content for item in window if item.type == "human"))
        self.assertLessEqual(sum(len(getattr(item, "content", "") or "") for item in window), 5000)

    def test_keeps_latest_user_and_workspace_when_tool_group_is_oversized(self) -> None:
        huge = "y" * 20000
        messages = [
            HumanMessage(content="old-user"),
            AIMessage(
                content="plan",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "b", "type": "tool_call"},
                    {"name": "read", "args": {}, "id": "a", "type": "tool_call"},
                ],
            ),
            ToolMessage(content="result-a", tool_call_id="a"),
            ToolMessage(content=huge, tool_call_id="b"),
            HumanMessage(content="最新问题：比较两家公司\n【工作环境指令】\n工作目录: output/session_demo"),
        ]
        window = build_context_window(messages, "base-system", max_chars=1200)
        text = "\n".join(getattr(item, "content", "") or "" for item in window)
        self.assertTrue(window)
        self.assertIn("最新问题", text)
        self.assertIn("【工作环境指令】", text)
        self.assertIn("output/session_demo", text)
        self.assertNotIn("old-user", text)
        self.assertIn("消息已按预算截断", text)
        self.assertLessEqual(sum(len(getattr(item, "content", "") or "") for item in window), 1200)
        ids = [
            str(getattr(item, "tool_call_id", "") or "")
            for item in window
            if item.type == "tool"
        ]
        self.assertEqual(set(ids), {"a", "b"})

    def test_middleware_counts_system_without_dropping_latest_user(self) -> None:
        request = type("Request", (), {})()
        request.messages = [
            HumanMessage(content="历史问题" + ("h" * 3000)),
            HumanMessage(content="当前问题请生成报告"),
        ]
        request.system_message = SystemMessage(content="系统契约" + ("s" * 500))
        request.override = lambda **kwargs: kwargs
        rewritten = ContextWindowMiddleware(max_chars=800)._rewrite(request)
        body_text = "\n".join(item.content for item in rewritten["messages"])
        self.assertIn("当前问题", body_text)
        self.assertNotIn("历史问题", body_text)
        self.assertLessEqual(len(rewritten["system_message"].content) + len(body_text), 800)

    def test_many_user_turns_do_not_grow_without_bound(self) -> None:
        messages = [HumanMessage(content=f"user-{index}-" + ("z" * 800)) for index in range(30)]
        window = build_context_window(messages, "system-" + ("s" * 200), max_chars=2400)
        total = sum(len(getattr(item, "content", "") or "") for item in window)
        self.assertLessEqual(total, 2400)
        self.assertLess(len(window), len(messages))
        self.assertTrue(window[-1].content.startswith("user-29"))

    def test_drops_unpaired_tool_and_incomplete_ai_calls(self) -> None:
        orphan = ToolMessage(content="lost", tool_call_id="missing")
        incomplete = AIMessage(
            content="",
            tool_calls=[{"name": "read", "args": {}, "id": "call-2", "type": "tool_call"}],
        )
        kept = HumanMessage(content="keep-me")
        paired = pair_tool_messages([orphan, incomplete, kept])
        self.assertEqual([item.content for item in paired], ["keep-me"])


class SqlGuardTests(unittest.TestCase):
    def test_allows_single_select_and_rejects_stacked_writes(self) -> None:
        safe = _assert_readonly("SELECT id FROM drugs WHERE name = 'a'")
        self.assertIn("SELECT", safe.upper())
        for unsafe in (
            "SELECT 1; DROP TABLE drugs",
            "UPDATE drugs SET name='x'",
            "SELECT 1; INSERT INTO drugs VALUES (1)",
            "/* comment */ DELETE FROM drugs",
            "",
        ):
            with self.assertRaises(ValueError):
                _assert_readonly(unsafe)


class ApiAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        get_settings.cache_clear()
        self._old = os.environ.get("ACCESS_TOKEN")
        os.environ["ACCESS_TOKEN"] = "demo-token"
        get_settings.cache_clear()
        server.settings = get_settings()

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("ACCESS_TOKEN", None)
        else:
            os.environ["ACCESS_TOKEN"] = self._old
        get_settings.cache_clear()
        server.settings = get_settings()

    def test_http_and_download_do_not_accept_token_query(self) -> None:
        with patch.object(server, "run_deep_agent", return_value=None), TestClient(server.app) as client:
            denied = client.post("/api/task", json={"query": "hi", "thread_id": "thread-demo"})
            self.assertEqual(denied.status_code, 401)

            allowed = client.post(
                "/api/task",
                json={"query": "hi", "thread_id": "thread-demo"},
                headers={"Authorization": "Bearer demo-token"},
            )
            self.assertEqual(allowed.status_code, 200)

            bad_id = client.post(
                "/api/task",
                json={"query": "hi", "thread_id": "../escape"},
                headers={"Authorization": "Bearer demo-token"},
            )
            self.assertEqual(bad_id.status_code, 422)

            leaked = client.get("/api/download", params={"thread_id": "thread-demo", "path": "a.md", "token": "demo-token"})
            self.assertEqual(leaked.status_code, 401)

            bad_thread = client.get(
                "/api/download",
                params={"thread_id": "../escape", "path": "a.md", "ticket": "1.deadbeef"},
            )
            self.assertEqual(bad_thread.status_code, 400)

            listed = client.get(
                "/api/files",
                params={"thread_id": "thread-demo"},
                headers={"Authorization": "Bearer demo-token"},
            )
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["files"], [])

            with patch.object(server, "get_pending_approval", return_value=[{"tool": "convert_md_to_pdf", "args": {}, "description": "preview"}]):
                status = client.get(
                    "/api/task/thread-demo/status",
                    headers={"Authorization": "Bearer demo-token"},
                )
            self.assertEqual(status.status_code, 200)
            body = status.json()
            self.assertTrue(body["awaiting_approval"])
            self.assertEqual(body["persistence"], "approval_checkpoint_only")
            self.assertEqual(body["actions"][0]["tool"], "convert_md_to_pdf")

            with patch.object(server, "get_pending_approval", return_value=[{"tool": "convert_md_to_pdf", "args": {}, "description": "preview"}]), patch.object(server, "resume_deep_agent", return_value=None) as resume_mock:
                rejected = client.post(
                    "/api/task/thread-demo/resume",
                    json={"decision": "reject", "feedback": "请缩短第二章"},
                    headers={"Authorization": "Bearer demo-token"},
                )
                self.assertEqual(rejected.status_code, 200)
                self.assertEqual(rejected.json()["decision"], "reject")
                resume_mock.assert_called()
                self.assertEqual(resume_mock.call_args.kwargs["feedback"], "请缩短第二章")

            approved_by_mistake = client.post(
                "/api/task/thread-demo/resume",
                json={"decision": "approve"},
                headers={"Authorization": "Bearer demo-token"},
            )
            self.assertEqual(approved_by_mistake.status_code, 409)

            missing = client.post(
                "/api/task/thread-demo/resume",
                json={"decision": "reject", "feedback": "无挂起任务"},
                headers={"Authorization": "Bearer demo-token"},
            )
            self.assertEqual(missing.status_code, 409)

    def test_websocket_requires_same_demo_token(self) -> None:
        with TestClient(server.app) as client:
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws/thread-demo"):
                    pass
            with client.websocket_connect("/ws/thread-demo?token=demo-token") as socket:
                socket.send_text("ping")
                payload = socket.receive_json()
                self.assertEqual(payload["type"], "pong")


if __name__ == "__main__":
    unittest.main()
