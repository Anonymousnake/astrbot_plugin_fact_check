from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot.api.message_components import Image, Plain, Reply
from astrbot_plugin_fact_check.fact_check import ClaimCandidate, FactCheckRequest, FactCheckResult
from astrbot_plugin_fact_check import main


class FakeBot:
    def __init__(self, *, fail_call_number: int | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_call_number = fail_call_number

    async def call_action(self, action: str, **payload):
        self.calls.append((action, payload))
        if self.fail_call_number == len(self.calls):
            raise RuntimeError("onebot send failed once")
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
    }
    plugin._reply_cache = {}
    plugin._fact_check_sessions = {}
    plugin._fact_check_tasks = set()
    plugin._active_followup_jobs = 0
    plugin._cooldown_until = 0.0
    plugin._fact_check_semaphore = main.asyncio.Semaphore(1)
    plugin._dump_forward_failure = lambda *args, **kwargs: None
    return plugin


class MainExperienceTests(unittest.IsolatedAsyncioTestCase):
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

        self.assertEqual(len(event.sent), 2)
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

    def test_cached_result_preserves_candidates_and_sources_for_followup_session(self) -> None:
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
        self.assertEqual([item.claim for item in session.candidates], ["请核查：A 事件是否属实？"])
        self.assertEqual(session.sources, ["https://example.com/source"])

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
            patch.object(plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)),
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
            patch.object(plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)),
            patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None),
            patch("astrbot_plugin_fact_check.main.run_fact_check_followup") as followup,
        ):
            await plugin.fact_check_followup(event)

        followup.assert_not_called()
        self.assertEqual(len(event.sent), 1)
        self.assertIn("队列满", event.sent[0]["plain"])

    async def test_onebot_text_retry_resumes_failed_chunk_without_duplicates(self) -> None:
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
        sent_chunks = [call[1]["message"][0]["data"]["text"] for call in event.bot.calls]
        self.assertEqual(sent_chunks, ["A" * 350, "A" * 10, "A" * 10])

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

    async def test_expired_fact_check_reply_followup_gets_explicit_message(self) -> None:
        plugin = make_plugin()
        event = FakeEvent(
            message_str="还能展开说说吗",
            messages=[Reply(id="1", message_str="事实核查：可信\n核查ID：fc_deadbeef")],
            fail_send=False,
        )

        with (
            patch.object(plugin, "_fetch_reply_payload", new=AsyncMock(return_value=None)),
            patch("astrbot_plugin_fact_check.main.is_plugin_allowed", None),
        ):
            await plugin.fact_check_followup(event)

        self.assertTrue(event.stopped)
        self.assertEqual(len(event.sent), 1)
        self.assertIn("上下文已过期", event.sent[0]["plain"])

    async def test_image_inputs_reject_untrusted_schemes_and_private_http(self) -> None:
        plugin = make_plugin()

        images = [
            NoLocalImage(file="base64://QUJD", url="base64://QUJD"),
            NoLocalImage(file="file:///etc/passwd", url="file:///etc/passwd"),
            NoLocalImage(file="http://127.0.0.1/private.png", url="http://127.0.0.1/private.png"),
            NoLocalImage(file="https://example.com/public.png", url="https://example.com/public.png"),
        ]

        result = await plugin._image_inputs(images, remaining=4)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, "https://example.com/public.png")


if __name__ == "__main__":
    unittest.main()
