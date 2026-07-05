from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_fact_check import main


class FakeBot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_action(self, action: str, **payload):
        self.calls.append((action, payload))
        return {"status": "ok"}


class FakeEvent:
    def __init__(self) -> None:
        self.bot = FakeBot()
        self.sent: list[object] = []

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

    async def send(self, payload):
        self.sent.append(payload)
        raise RuntimeError("forward send failed")


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


if __name__ == "__main__":
    unittest.main()
