from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot.api.message_components import Image, Plain, Reply
from astrbot_plugin_fact_check import main
from astrbot_plugin_fact_check.fact_check import (
    ClaimCandidate,
    FactCheckRequest,
    FactCheckResult,
    ImageInput,
)
from astrbot_plugin_fact_check.runtime import AsyncSingleFlight
from astrbot_plugin_fact_check.storage import FactCheckMetricsStore


class FakeBot:
    def __init__(self, *, fail_call_number: int | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_call_number = fail_call_number

    async def call_action(self, action: str, **payload):
        self.calls.append((action, payload))
        if self.fail_call_number == len(self.calls):
            raise RuntimeError("onebot send failed once")
        return {"status": "ok"}


class ConfirmTimeoutBot(FakeBot):
    def __init__(self, *, timeout_call_number: int) -> None:
        super().__init__()
        self.timeout_call_number = timeout_call_number

    async def call_action(self, action: str, **payload):
        self.calls.append((action, payload))
        if self.timeout_call_number == len(self.calls):
            raise RuntimeError("confirm timeout")
        return {"status": "ok"}


class FakeEvent:
    def __init__(
        self,
        *,
        message_str: str = "",
        messages: list[object] | None = None,
        fail_send: bool = True,
        bot: FakeBot | None = None,
    ) -> None:
        self.message_str = message_str
        self._messages = messages if messages is not None else []
        self.bot = bot or FakeBot()
        self.sent: list[object] = []
        self.extras: dict[str, object] = {}
        self.stopped = False
        self.fail_send = fail_send

    def get_group_id(self):
        return "123456"

    def get_sender_id(self):
        return "654321"

    def get_self_id(self):
        return "111111"

    def chain_result(self, payload):
        return {"chain": payload}

    def plain_result(self, text: str):
        return {"plain": text}

    def get_message_str(self):
        return self.message_str

    def get_messages(self):
        return self._messages

    def set_extra(self, key: str, value):
        self.extras[key] = value

    def stop_event(self):
        self.stopped = True

    async def send(self, payload):
        self.sent.append(payload)
        if self.fail_send:
            raise RuntimeError("forward send failed")


class NoLocalImage(Image):
    async def convert_to_file_path(self):
        raise RuntimeError("no local file")


class CountingImage(Image):
    resolve_calls: int = 0

    def __init__(self, *, file: str, url: str) -> None:
        super().__init__(file=file, url=url)
        self.resolve_calls = 0

    async def convert_to_file_path(self):
        self.resolve_calls += 1
        raise RuntimeError("no local file")


def make_plugin() -> main.FactCheckPlugin:
    plugin = object.__new__(main.FactCheckPlugin)
    plugin.config = {
        "enable_fact_check": True,
        "fact_check_cache_ttl_seconds": 600,
        "fact_check_cache_max_entries": 32,
        "fact_check_followup_ttl_seconds": 3600,
        "fact_check_followup_max_sessions": 50,
        "fact_check_max_queue": 1,
        "fact_check_total_timeout_seconds": 30,
        "fact_check_image_download_timeout_seconds": 1,
        "fact_check_max_images": 3,
        # Most experience tests isolate behavior unrelated to the optional ACL
        # plugin. Production defaults remain fail-closed when ACL is missing.
        "fact_check_access_control_fail_open": True,
    }
    plugin._reply_cache = {}
    plugin._fact_check_sessions = {}
    plugin._fact_check_tasks = set()
    plugin._singleflight = AsyncSingleFlight()
    plugin._active_followup_jobs = 0
    plugin._cooldown_until = 0.0
    plugin._fact_check_semaphore = main.asyncio.Semaphore(1)
    plugin._session_store_enabled = False
    plugin._metrics_store = FactCheckMetricsStore(None)
    plugin._dump_forward_failure = lambda *args, **kwargs: None
    return plugin


class MainExperienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_outcome_is_recorded_separately_from_pipeline_success(
        self,
    ) -> None:
        plugin = make_plugin()
        event = FakeEvent(fail_send=False)

        with patch.object(
            plugin, "_send_fact_check_reply_impl", new=AsyncMock(return_value=False)
        ):
            sent = await plugin._send_fact_check_reply(
                event,
                "事实核查：可信",
                label="test",
                purpose="result",
            )

        self.assertFalse(sent)
        self.assertEqual(plugin._metrics_store.snapshot()["delivery_failure"], 1)

    async def test_duplicate_inflight_jobs_share_one_pipeline_call(self) -> None:
        plugin = make_plugin()
        request = FactCheckRequest(text="A 事件", trigger_text="/事实核查")
        first_event = FakeEvent(fail_send=False)
        second_event = FakeEvent(fail_send=False)
        started = threading.Event()
        release = threading.Event()

        def run_once(*_args):
            started.set()
            release.wait(timeout=2)
            return FactCheckResult("事实核查：可信", "ok")

        with (
            patch.object(plugin, "_run_fact_check_sync", side_effect=run_once) as run,
            patch.object(plugin, "_send_fact_check_reply", new=AsyncMock()) as send,
        ):
            first = asyncio.create_task(
                plugin._run_fact_check_job(
                    first_event, request, time.perf_counter(), "same-key"
                ),
            )
            await asyncio.to_thread(started.wait, 1)
            second = asyncio.create_task(
                plugin._run_fact_check_job(
                    second_event, request, time.perf_counter(), "same-key"
                ),
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, second)

        self.assertEqual(run.call_count, 1)
        self.assertEqual(send.await_count, 2)

    def test_cache_key_changes_when_pipeline_version_changes(self) -> None:
        plugin = make_plugin()
        request = FactCheckRequest(text="A 事件", trigger_text="/事实核查")

        with patch("astrbot_plugin_fact_check.main.FACT_CHECK_PIPELINE_VERSION", "v1"):
            first = plugin._request_cache_key(request)
        with patch("astrbot_plugin_fact_check.main.FACT_CHECK_PIPELINE_VERSION", "v2"):
            second = plugin._request_cache_key(request)

        self.assertNotEqual(first, second)

    def test_breaking_news_uses_a_shorter_cache_ttl(self) -> None:
        plugin = make_plugin()

        self.assertEqual(
            plugin._cache_ttl_for_request(
                FactCheckRequest("今天最新突发消息", "/事实核查")
            ),
            120,
        )
        self.assertEqual(
            plugin._cache_ttl_for_request(
                FactCheckRequest("历史人物出生年份", "/事实核查")
            ),
            600,
        )

    async def test_status_command_reports_persisted_aggregate_metrics(self) -> None:
        plugin = make_plugin()
        plugin._metrics_store.record(outcome="partial", elapsed=2.0)
        event = FakeEvent(fail_send=False)

        with patch(
            "astrbot_plugin_fact_check.main.is_plugin_allowed", return_value=True
        ):
            results = [item async for item in plugin.factcheck_status(event)]

        self.assertTrue(event.stopped)
        self.assertEqual(len(results), 1)
        self.assertIn("部分完成：1", results[0]["plain"])

    def test_group_followup_session_is_visible_only_to_its_owner(self) -> None:
        session = main.FactCheckSession(
            session_id="fc_abcd1234",
            created_at=time.time(),
            group_id="123456",
            user_id="owner-user",
            request_data=FactCheckRequest(text="A 事件", trigger_text="/事实核查"),
            reply="事实核查：可信",
        )
        owner_event = SimpleNamespace(
            get_group_id=lambda: "123456",
            get_sender_id=lambda: "owner-user",
        )
        other_member_event = SimpleNamespace(
            get_group_id=lambda: "123456",
            get_sender_id=lambda: "other-user",
        )

        self.assertTrue(
            main.FactCheckPlugin._session_visible_to_event(session, owner_event)
        )
        self.assertFalse(
            main.FactCheckPlugin._session_visible_to_event(session, other_member_event)
        )

    def test_missing_access_control_fails_closed_by_default(self) -> None:
        plugin = make_plugin()
        plugin.config["fact_check_access_control_fail_open"] = False

        with patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None):
            self.assertFalse(plugin._is_fact_check_allowed(FakeEvent()))

    def test_qq_fallback_does_not_rewrite_fact_check_terms(self) -> None:
        plugin = make_plugin()
        text = "网信办回应 VPN 与 Reddit 相关说法。"

        self.assertEqual(plugin._sanitize_forward_text_for_qq(text), text)

    async def test_failed_fact_check_is_not_cached_or_saved_for_followup(self) -> None:
        plugin = make_plugin()
        event = FakeEvent(fail_send=False)
        request = FactCheckRequest(text="A 事件", trigger_text="/事实核查")
        failed = FactCheckResult(reply="这条我现在没查成。", reason="request timeout")

        with (
            patch("astrbot_plugin_fact_check.main.run_fact_check", return_value=failed),
            patch.object(plugin, "_send_fact_check_reply", new=AsyncMock()) as send,
        ):
            await plugin._run_fact_check_job(
                event, request, time.perf_counter(), "cache-key"
            )

        self.assertEqual(plugin._reply_cache, {})
        self.assertEqual(plugin._fact_check_sessions, {})
        self.assertIsNone(send.call_args.kwargs["session_id"])

    def test_request_cache_key_uses_image_content_instead_of_snapshot_path(
        self,
    ) -> None:
        plugin = make_plugin()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same-image")
            second.write_bytes(b"same-image")

            first_key = plugin._request_cache_key(
                FactCheckRequest(
                    "", "/事实核查", images=[ImageInput("", "first.png", str(first))]
                ),
            )
            second_key = plugin._request_cache_key(
                FactCheckRequest(
                    "", "/事实核查", images=[ImageInput("", "second.png", str(second))]
                ),
            )

        self.assertEqual(first_key, second_key)

    def test_request_cache_key_changes_for_different_image_content(self) -> None:
        plugin = make_plugin()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"first-image")
            second.write_bytes(b"second-image")

            first_key = plugin._request_cache_key(
                FactCheckRequest(
                    "", "/事实核查", images=[ImageInput("", "first.png", str(first))]
                ),
            )
            second_key = plugin._request_cache_key(
                FactCheckRequest(
                    "", "/事实核查", images=[ImageInput("", "second.png", str(second))]
                ),
            )

        self.assertNotEqual(first_key, second_key)

    def test_duplicate_image_inputs_are_collapsed_by_content(self) -> None:
        plugin = make_plugin()
        images = [
            ImageInput("https://example.com/a.png", content_sha256="abc"),
            ImageInput("https://example.com/b.png", content_sha256="abc"),
            ImageInput("https://example.com/c.png", content_sha256="def"),
        ]

        result = plugin._dedupe_image_inputs(images)

        self.assertEqual([item.content_sha256 for item in result], ["abc", "def"])

    async def test_send_fact_check_reply_falls_back_to_onebot_text(self) -> None:
        plugin = object.__new__(main.FactCheckPlugin)
        plugin._dump_forward_failure = lambda *args, **kwargs: None
        event = FakeEvent()
        original_send_chain_result = main.send_chain_result
        original_sleep = main.asyncio.sleep

        async def fake_sleep(_seconds):
            return None

        main.send_chain_result = None
        main.asyncio.sleep = fake_sleep
        try:
            await plugin._send_fact_check_reply(
                event,
                "事实核查：可信",
                label="test",
                purpose="result",
                session_id="fc_1234abcd",
            )
        finally:
            main.send_chain_result = original_send_chain_result
            main.asyncio.sleep = original_sleep

        self.assertEqual(len(event.sent), 1)
        self.assertEqual(event.bot.calls[0][0], "send_msg")
        sent_text = event.bot.calls[0][1]["message"][0]["data"]["text"]
        self.assertIn("事实核查：可信", sent_text)
        self.assertIn("核查ID：fc_1234abcd", sent_text)

    def test_session_marker_helper_appends_id_for_text_fallback(self) -> None:
        text = main.FactCheckPlugin._fact_check_text_with_session_marker(
            "事实核查：可信",
            session_id="fc_1234abcd",
        )

        self.assertIn("追问可回复本消息", text)
        self.assertIn("核查ID：fc_1234abcd", text)

    def test_public_forward_test_command_is_removed(self) -> None:
        self.assertFalse(hasattr(main.FactCheckPlugin, "fact_check_forward_test"))

    def test_cached_result_preserves_candidates_and_sources_for_followup_session(
        self,
    ) -> None:
        plugin = make_plugin()
        event = FakeEvent(fail_send=False)
        request = FactCheckRequest(text="A 事件是真的", trigger_text="/事实核查")
        result = FactCheckResult(
            reply="事实核查：可信",
            reason="ok",
            sources=["https://example.com/source"],
            candidates=[ClaimCandidate("请核查：A 事件是否属实？", "用户文字", 5)],
        )

        plugin._set_cached_result("cache-key", result)
        cached = plugin._get_cached_result("cache-key")
        self.assertIsNotNone(cached)
        session_id = plugin._remember_fact_check_session(event, request, cached)

        session = plugin._fact_check_sessions[session_id]
        self.assertEqual(
            [item.claim for item in session.candidates], ["请核查：A 事件是否属实？"]
        )
        self.assertEqual(session.sources, ["https://example.com/source"])

    def test_followup_sessions_persist_without_original_media_or_text(self) -> None:
        event = FakeEvent(fail_send=False)
        result = FactCheckResult(
            reply="事实核查：部分存疑",
            reason="ok",
            sources=["https://example.com/report"],
            candidates=[ClaimCandidate("请核查：A 事件是否属实？", "图片", 5)],
        )
        request = FactCheckRequest(
            text="含敏感原文",
            trigger_text="/事实核查",
            images=[
                ImageInput(
                    url="https://example.com/private.png?token=secret",
                    path="C:/temp/private.png",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plugin = make_plugin()
            plugin._session_store_enabled = True
            with patch.object(main.StarTools, "get_data_dir", return_value=root):
                session_id = plugin._remember_fact_check_session(event, request, result)
                raw = (root / "fact_check_sessions.json").read_text(encoding="utf-8")

                restored = make_plugin()
                restored._session_store_enabled = True
                restored._load_fact_check_sessions()

        self.assertNotIn("含敏感原文", raw)
        self.assertNotIn("private.png", raw)
        self.assertNotIn("secret", raw)
        self.assertIn(session_id, restored._fact_check_sessions)
        session = restored._fact_check_sessions[session_id]
        self.assertEqual(session.request_data.text, "")
        self.assertEqual(session.request_data.images, [])
        self.assertEqual(session.candidates[0].claim, "请核查：A 事件是否属实？")

    def test_v1_session_store_loads_and_is_rewritten_as_v2(self) -> None:
        created_at = time.time() - 10
        payload = {
            "version": 1,
            "sessions": [
                {
                    "session_id": "fc_1234abcd",
                    "created_at": created_at,
                    "group_id": "123456",
                    "user_id": "654321",
                    "reply": "事实核查：证据不足",
                    "candidates": [],
                    "sources": [],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = root / "fact_check_sessions.json"
            store.write_text(json.dumps(payload), encoding="utf-8")
            plugin = make_plugin()
            plugin._session_store_enabled = True
            with patch.object(main.StarTools, "get_data_dir", return_value=root):
                plugin._load_fact_check_sessions()
                session = plugin._fact_check_sessions["fc_1234abcd"]
                plugin._persist_fact_check_sessions()
                rewritten = json.loads(store.read_text(encoding="utf-8"))

        self.assertEqual(session.updated_at, created_at)
        self.assertEqual(rewritten["version"], 2)
        self.assertEqual(rewritten["sessions"][0]["updated_at"], created_at)

    def test_forward_failure_dump_omits_reply_text_by_default(self) -> None:
        plugin = object.__new__(main.FactCheckPlugin)
        plugin.config = {"fact_check_debug_store_full_failure_text": False}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(main.StarTools, "get_data_dir", return_value=root):
                plugin._dump_forward_failure(
                    "group:123:user:456",
                    "敏感事实核查正文 https://example.com/?token=secret",
                    RuntimeError(
                        "send failed for https://example.com/image.png?token=secret "
                        "at C:\\Users\\test\\private.png",
                    ),
                )
            payload = json.loads(
                (root / "last_forward_failure.json").read_text(encoding="utf-8"),
            )

        self.assertNotIn("text", payload)
        self.assertNotIn("chunks", payload)
        self.assertEqual(
            payload["length"], len("敏感事实核查正文 https://example.com/?token=secret")
        )
        self.assertIn("text_sha256", payload)
        self.assertNotIn("secret", payload["error"])
        self.assertNotIn("private.png", payload["error"])

    def test_model_cooling_error_starts_user_facing_cooldown(self) -> None:
        plugin = make_plugin()
        plugin.config["fact_check_rate_limit_cooldown_seconds"] = 90

        plugin._maybe_start_cooldown(
            "all configured fact-check models are cooling down: gemini-2.5-flash",
        )

        self.assertGreater(plugin._cooldown_left(), 80)

    def test_fact_check_metrics_log_aggregate_without_request_content(self) -> None:
        plugin = make_plugin()
        plugin.config["fact_check_metrics_log_every"] = 3

        with patch.object(main.logger, "info") as info:
            plugin._record_fact_check_metric(success=True, elapsed=2.0)
            plugin._record_fact_check_metric(
                success=True,
                elapsed=0.1,
                cache_hit=True,
            )
            plugin._record_fact_check_metric(
                success=False,
                elapsed=4.0,
                followup=True,
            )

        logs = "\n".join(str(call.args[0]) for call in info.call_args_list if call.args)
        self.assertIn("requests=3", logs)
        self.assertIn("pipeline_success=2", logs)
        self.assertIn("pipeline_failure=1", logs)
        self.assertIn("cache_hits=1", logs)
        self.assertIn("followups=1", logs)
        self.assertNotIn("request_data", logs)

    async def test_followup_respects_cooldown_before_progress_message(self) -> None:
        plugin = make_plugin()
        plugin._cooldown_until = time.time() + 30
        request = FactCheckRequest(text="A 事件是真的", trigger_text="/事实核查")
        plugin._fact_check_sessions["fc_aaaabbbb"] = main.FactCheckSession(
            session_id="fc_aaaabbbb",
            created_at=time.time(),
            group_id="123456",
            user_id="654321",
            request_data=request,
            reply="事实核查：可信",
            candidates=[ClaimCandidate("请核查：A 事件是否属实？")],
            sources=[],
        )
        event = FakeEvent(
            message_str="还能展开说说吗",
            messages=[Reply(id="1", message_str="事实核查：可信\n核查ID：fc_aaaabbbb")],
            fail_send=False,
        )

        with (
            patch.object(
                plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)
            ),
            patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None),
            patch("astrbot_plugin_fact_check.main.run_fact_check_followup") as followup,
        ):
            await plugin.fact_check_followup(event)

        followup.assert_not_called()
        self.assertEqual(len(event.sent), 1)
        self.assertIn("冷却", event.sent[0]["plain"])
        self.assertNotIn("我接着查一下", event.sent[0]["plain"])

    async def test_followup_respects_queue_limit_before_progress_message(self) -> None:
        plugin = make_plugin()
        plugin._fact_check_tasks = {object()}
        request = FactCheckRequest(text="A 事件是真的", trigger_text="/事实核查")
        plugin._fact_check_sessions["fc_bbbbcccc"] = main.FactCheckSession(
            session_id="fc_bbbbcccc",
            created_at=time.time(),
            group_id="123456",
            user_id="654321",
            request_data=request,
            reply="事实核查：可信",
            candidates=[ClaimCandidate("请核查：A 事件是否属实？")],
            sources=[],
        )
        event = FakeEvent(
            message_str="还能展开说说吗",
            messages=[Reply(id="1", message_str="事实核查：可信\n核查ID：fc_bbbbcccc")],
            fail_send=False,
        )

        with (
            patch.object(
                plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)
            ),
            patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None),
            patch("astrbot_plugin_fact_check.main.run_fact_check_followup") as followup,
        ):
            await plugin.fact_check_followup(event)

        followup.assert_not_called()
        self.assertEqual(len(event.sent), 1)
        self.assertIn("队列满", event.sent[0]["plain"])

    async def test_successful_followup_advances_and_persists_session(self) -> None:
        plugin = make_plugin()
        request = FactCheckRequest(text="A 事件是真的", trigger_text="/事实核查")
        session = main.FactCheckSession(
            session_id="fc_abcd1234",
            created_at=time.time() - 30,
            group_id="123456",
            user_id="654321",
            request_data=request,
            reply="事实核查：部分存疑",
            candidates=[ClaimCandidate("请核查：A 事件是否属实？")],
            sources=["旧来源：https://example.com/old"],
        )
        plugin._fact_check_sessions[session.session_id] = session
        event = FakeEvent(
            message_str="新证据会改变结论吗",
            messages=[
                Reply(
                    id="1",
                    message_str=f"{session.reply}\n核查ID：{session.session_id}",
                ),
            ],
            fail_send=False,
        )
        followup_result = FactCheckResult(
            reply=(
                "追问结论：新证据成立。\n"
                "补充依据：官方来源确认。\n"
                "是否改变原结论：原结论需要修正\n"
                "来源：官方来源"
            ),
            reason="ok; follow-up",
            sources=["官方来源：https://example.gov/new"],
            candidates=session.candidates,
        )

        with (
            patch.object(
                plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)
            ),
            patch.object(
                plugin, "_send_fact_check_reply", new=AsyncMock(return_value=True)
            ),
            patch.object(plugin, "_persist_fact_check_sessions") as persist,
            patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None),
            patch(
                "astrbot_plugin_fact_check.main.run_fact_check_followup",
                return_value=followup_result,
            ) as followup,
        ):
            await plugin.fact_check_followup(event)

        self.assertEqual(session.reply, followup_result.reply)
        self.assertEqual(
            session.sources,
            [
                "官方来源：https://example.gov/new",
                "旧来源：https://example.com/old",
            ],
        )
        self.assertGreater(session.updated_at, session.created_at)
        persist.assert_called_once()
        self.assertEqual(
            followup.call_args.kwargs["previous_reply"],
            "事实核查：部分存疑",
        )

    async def test_failed_followup_send_does_not_advance_session(self) -> None:
        plugin = make_plugin()
        session = main.FactCheckSession(
            session_id="fc_dcba4321",
            created_at=time.time(),
            group_id="123456",
            user_id="654321",
            request_data=FactCheckRequest("A 事件", "/事实核查"),
            reply="事实核查：证据不足",
            sources=["https://example.com/old"],
        )
        plugin._fact_check_sessions[session.session_id] = session
        event = FakeEvent(
            message_str="再查一下",
            messages=[
                Reply(
                    id="1", message_str=f"{session.reply}\n核查ID：{session.session_id}"
                )
            ],
            fail_send=False,
        )

        with (
            patch.object(
                plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)
            ),
            patch.object(
                plugin, "_send_fact_check_reply", new=AsyncMock(return_value=False)
            ),
            patch.object(plugin, "_persist_fact_check_sessions") as persist,
            patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None),
            patch(
                "astrbot_plugin_fact_check.main.run_fact_check_followup",
                return_value=FactCheckResult(
                    reply=(
                        "追问结论：新信息。\n"
                        "补充依据：新依据。\n"
                        "是否改变原结论：原结论暂不改变\n"
                        "来源：新来源"
                    ),
                    reason="ok; follow-up",
                    sources=["https://example.com/new"],
                ),
            ),
        ):
            await plugin.fact_check_followup(event)

        self.assertEqual(session.reply, "事实核查：证据不足")
        self.assertEqual(session.sources, ["https://example.com/old"])
        persist.assert_not_called()

    async def test_onebot_text_retry_resumes_failed_chunk_without_duplicates(
        self,
    ) -> None:
        plugin = make_plugin()
        event = FakeEvent(fail_send=False, bot=FakeBot(fail_call_number=2))
        original_sleep = main.asyncio.sleep

        async def fake_sleep(_seconds):
            return None

        main.asyncio.sleep = fake_sleep
        try:
            ok = await plugin._send_text_via_onebot(
                event,
                "A" * 360,
                label="retry-test",
                prefer_send_msg=True,
            )
        finally:
            main.asyncio.sleep = original_sleep

        self.assertTrue(ok)
        sent_chunks = [
            call[1]["message"][0]["data"]["text"] for call in event.bot.calls
        ]
        self.assertEqual(sent_chunks, ["A" * 350, "A" * 10, "A" * 10])

    async def test_onebot_confirm_timeout_continues_with_remaining_chunks(self) -> None:
        plugin = make_plugin()
        event = FakeEvent(fail_send=False, bot=ConfirmTimeoutBot(timeout_call_number=1))

        with (
            patch(
                "astrbot_plugin_fact_check.main.is_confirm_timeout", return_value=True
            ),
            patch("astrbot_plugin_fact_check.main.asyncio.sleep", new=AsyncMock()),
        ):
            ok = await plugin._send_text_via_onebot(
                event,
                "A" * 360,
                label="confirm-timeout-test",
                prefer_send_msg=True,
            )

        self.assertTrue(ok)
        sent_chunks = [
            call[1]["message"][0]["data"]["text"] for call in event.bot.calls
        ]
        self.assertEqual(sent_chunks, ["A" * 350, "A" * 10])

    async def test_forward_rejection_skips_identical_forward_retry(self) -> None:
        plugin = make_plugin()
        event = FakeEvent(fail_send=False)
        rejected = SimpleNamespace(
            ok=False,
            assumed_ok=False,
            kind="forward_rejected",
            error="rejected",
        )

        with (
            patch(
                "astrbot_plugin_fact_check.main.send_chain_result",
                new=AsyncMock(return_value=rejected),
            ) as send_forward,
            patch("astrbot_plugin_fact_check.main.asyncio.sleep", new=AsyncMock()),
        ):
            await plugin._send_fact_check_reply(
                event,
                "事实核查：可信",
                label="forward-rejection-test",
                purpose="result",
            )

        self.assertEqual(send_forward.await_count, 1)
        self.assertEqual(len(event.bot.calls), 1)
        self.assertEqual(event.bot.calls[0][0], "send_msg")

    async def test_bare_chinese_fact_check_command_returns_usage(self) -> None:
        plugin = make_plugin()
        event = FakeEvent(
            message_str="/事实核查",
            messages=[Plain("/事实核查")],
            fail_send=False,
        )
        outputs = []

        with patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None):
            async for item in plugin.fact_check(event):
                outputs.append(item)

        self.assertEqual(len(outputs), 1)
        self.assertIn("用法：回复一条消息后发送 /事实核查", outputs[0]["plain"])

    async def test_expired_fact_check_reply_followup_gets_explicit_message(
        self,
    ) -> None:
        plugin = make_plugin()
        event = FakeEvent(
            message_str="还能展开说说吗",
            messages=[Reply(id="1", message_str="事实核查：可信\n核查ID：fc_deadbeef")],
            fail_send=False,
        )

        with (
            patch.object(
                plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)
            ),
            patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None),
        ):
            await plugin.fact_check_followup(event)

        self.assertTrue(event.stopped)
        self.assertEqual(len(event.sent), 1)
        self.assertIn("上下文已过期", event.sent[0]["plain"])

    async def test_unrelated_reply_without_sessions_does_not_fetch_remote_payload(
        self,
    ) -> None:
        plugin = make_plugin()
        event = FakeEvent(
            message_str="这是什么意思",
            messages=[Reply(id="123", message_str="普通聊天内容")],
            fail_send=False,
        )

        with patch.object(plugin, "_fetch_reply_payload", new=AsyncMock()) as fetch:
            session, missing = await plugin._find_followup_session_with_state(event)

        self.assertIsNone(session)
        self.assertFalse(missing)
        fetch.assert_not_awaited()

    async def test_local_fact_check_marker_matches_session_without_remote_fetch(
        self,
    ) -> None:
        plugin = make_plugin()
        session = main.FactCheckSession(
            session_id="fc_aaaabbbb",
            created_at=time.time(),
            group_id="123456",
            user_id="654321",
            request_data=FactCheckRequest("A 事件", "/事实核查"),
            reply="事实核查：可信\n1. 核查点：A 事件。\n结论：已核实。",
        )
        plugin._fact_check_sessions[session.session_id] = session
        event = FakeEvent(
            message_str="再解释一下",
            messages=[
                Reply(
                    id="123",
                    message_str=f"{session.reply}\n核查ID：{session.session_id}",
                ),
            ],
            fail_send=False,
        )

        with patch.object(plugin, "_fetch_reply_payload", new=AsyncMock()) as fetch:
            matched, missing = await plugin._find_followup_session_with_state(event)

        self.assertIs(matched, session)
        self.assertFalse(missing)
        fetch.assert_not_awaited()

    async def test_markerless_followup_matches_the_quoted_session_not_the_latest(
        self,
    ) -> None:
        plugin = make_plugin()
        older = main.FactCheckSession(
            session_id="fc_11111111",
            created_at=time.time() - 10,
            group_id="123456",
            user_id="654321",
            request_data=FactCheckRequest("older", "/事实核查"),
            reply="事实核查：部分存疑\n1. 核查点：旧事件是否发生。\n结论：证据不足。",
        )
        latest = main.FactCheckSession(
            session_id="fc_22222222",
            created_at=time.time(),
            group_id="123456",
            user_id="654321",
            request_data=FactCheckRequest("latest", "/事实核查"),
            reply="事实核查：可信\n1. 核查点：新事件已经发生。\n结论：已核实。",
        )
        plugin._fact_check_sessions = {
            older.session_id: older,
            latest.session_id: latest,
        }
        event = FakeEvent(
            message_str="再解释一下",
            messages=[Reply(id="1", message_str=older.reply)],
            fail_send=False,
        )

        with patch.object(
            plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)
        ):
            session, missing = await plugin._find_followup_session_with_state(event)

        self.assertFalse(missing)
        self.assertIs(session, older)

    async def test_ambiguous_markerless_followup_does_not_guess_latest_session(
        self,
    ) -> None:
        plugin = make_plugin()
        for index in range(2):
            session_id = f"fc_{index + 1:08d}"
            plugin._fact_check_sessions[session_id] = main.FactCheckSession(
                session_id=session_id,
                created_at=time.time() + index,
                group_id="123456",
                user_id="654321",
                request_data=FactCheckRequest(str(index), "/事实核查"),
                reply=f"事实核查：可信\n核查点：事件 {index}。",
            )
        event = FakeEvent(
            message_str="再解释一下",
            messages=[Reply(id="1", message_str="事实核查：可信")],
            fail_send=False,
        )

        with patch.object(
            plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)
        ):
            session, missing = await plugin._find_followup_session_with_state(event)

        self.assertIsNone(session)
        self.assertTrue(missing)

    async def test_image_inputs_reject_untrusted_schemes_and_private_http(self) -> None:
        plugin = make_plugin()

        images = [
            NoLocalImage(file="base64://QUJD", url="base64://QUJD"),
            NoLocalImage(file="file:///etc/passwd", url="file:///etc/passwd"),
            NoLocalImage(
                file="http://127.0.0.1/private.png", url="http://127.0.0.1/private.png"
            ),
            NoLocalImage(
                file="https://example.com/public.png",
                url="https://example.com/public.png",
            ),
        ]

        result = await plugin._image_inputs(images, remaining=4)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, "https://example.com/public.png")

    async def test_image_inputs_snapshot_existing_component_path(self) -> None:
        plugin = make_plugin()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "astrbot-temp.png"
            source.write_bytes(b"stable-image-content")
            component = Image(file="quoted.png", path=str(source))

            with patch.object(
                main.StarTools, "get_data_dir", return_value=root / "plugin-data"
            ):
                result = await plugin._image_inputs([component], remaining=1)

            self.assertEqual(len(result), 1)
            self.assertNotEqual(result[0].path, str(source))
            self.assertTrue(Path(result[0].path).is_file())
            source.unlink()
            self.assertEqual(Path(result[0].path).read_bytes(), b"stable-image-content")

    async def test_image_inputs_skip_duplicate_reference_before_resolution(
        self,
    ) -> None:
        plugin = make_plugin()
        first = CountingImage(
            file="same.png",
            url="https://example.com/same.png?token=secret",
        )
        duplicate = CountingImage(
            file="same.png",
            url="https://example.com/same.png?token=secret",
        )
        seen_refs: set[str] = set()

        first_result = await plugin._image_inputs(
            [first],
            remaining=2,
            seen_refs=seen_refs,
        )
        duplicate_result = await plugin._image_inputs(
            [duplicate],
            remaining=2,
            seen_refs=seen_refs,
        )

        self.assertEqual(len(first_result), 1)
        self.assertEqual(duplicate_result, [])
        self.assertEqual(first.resolve_calls, 1)
        self.assertEqual(duplicate.resolve_calls, 0)

    async def test_image_inputs_keep_same_filename_with_different_urls(self) -> None:
        plugin = make_plugin()
        first = CountingImage(file="image.png", url="https://a.example/image.png")
        second = CountingImage(file="image.png", url="https://b.example/image.png")
        seen_refs: set[str] = set()

        result = await plugin._image_inputs(
            [first, second],
            remaining=2,
            seen_refs=seen_refs,
        )

        self.assertEqual([item.url for item in result], [first.url, second.url])
        self.assertEqual(first.resolve_calls, 1)
        self.assertEqual(second.resolve_calls, 1)

    async def test_image_input_logs_hide_signed_url_and_local_path(self) -> None:
        plugin = make_plugin()
        component = CountingImage(
            file="https://gchat.qpic.cn/path/image.png?rkey=secret",
            url="https://gchat.qpic.cn/path/image.png?rkey=secret",
        )

        with (
            patch.object(main.logger, "info") as info,
            patch.object(main.logger, "warning") as warning,
        ):
            await plugin._image_inputs([component], remaining=1)

        logs = "\n".join(
            str(call.args[0])
            for call in [*info.call_args_list, *warning.call_args_list]
            if call.args
        )
        self.assertNotIn("secret", logs)
        self.assertNotIn("rkey", logs)
        self.assertNotIn("gchat.qpic.cn/path", logs)
        self.assertIn("image.png", logs)

    def test_image_cache_prunes_by_age_size_and_file_count(self) -> None:
        plugin = make_plugin()
        plugin.config.update(
            {
                "fact_check_image_cache_ttl_seconds": 60,
                "fact_check_image_cache_max_bytes": 12,
                "fact_check_image_cache_max_files": 2,
            },
        )
        plugin._image_cache_last_prune = 0.0
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            expired = cache_dir / "expired.img"
            oldest = cache_dir / "oldest.img"
            newest = cache_dir / "newest.img"
            extra = cache_dir / "extra.img"
            for item in (expired, oldest, newest, extra):
                item.write_bytes(b"123456")
            now = time.time()
            os.utime(expired, (now - 120, now - 120))
            os.utime(oldest, (now - 30, now - 30))
            os.utime(newest, (now - 20, now - 20))
            os.utime(extra, (now - 10, now - 10))

            plugin._prune_image_input_cache(cache_dir, force=True)

            self.assertFalse(expired.exists())
            self.assertFalse(oldest.exists())
            self.assertTrue(newest.exists())
            self.assertTrue(extra.exists())

    def test_image_cache_hit_refreshes_lru_timestamp(self) -> None:
        plugin = make_plugin()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            source.write_bytes(b"same-image")
            with patch.object(
                main.StarTools, "get_data_dir", return_value=root / "data"
            ):
                target = Path(
                    plugin._snapshot_image_path(str(source), file_name="source.png")
                )
                old_time = time.time() - 100
                os.utime(target, (old_time, old_time))
                refreshed = Path(
                    plugin._snapshot_image_path(str(source), file_name="source.png")
                )

            self.assertEqual(refreshed, target)
            self.assertGreater(refreshed.stat().st_mtime, old_time)


if __name__ == "__main__":
    unittest.main()
