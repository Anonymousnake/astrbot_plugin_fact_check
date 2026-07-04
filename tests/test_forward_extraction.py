from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for astrbot_root in (Path("D:/Codex/AstrBot"), Path("/home/ubuntu/AstrBot")):
    if astrbot_root.exists():
        sys.path.insert(0, str(astrbot_root))

from astrbot.api.message_components import Forward, Reply
from astrbot_plugin_fact_check.main import FactCheckPlugin


class FakeForwardClient:
    calls: list[str] = []

    def __init__(self, event, *args, **kwargs) -> None:
        self.event = event

    async def get_forward_msg(self, forward_id):
        self.__class__.calls.append(str(forward_id))
        return {"data": {"messages": []}}


class FakeForwardParser:
    def parse_get_forward_payload(self, payload):
        return {
            "text": "老破B: like\n动画表情: 岩浆垃圾桶\n食客: like",
            "image_refs": [],
            "forward_ids": [],
        }


class FakeEvent:
    message_str = "/事实核查"

    def __init__(self, messages) -> None:
        self._messages = messages

    def get_message_str(self):
        return self.message_str

    def get_messages(self):
        return self._messages

    def get_group_id(self):
        return "951944306"

    def get_sender_id(self):
        return "462695210"


def make_plugin() -> FactCheckPlugin:
    plugin = FactCheckPlugin.__new__(FactCheckPlugin)
    plugin.config = {
        "fact_check_max_images": 2,
        "fact_check_image_download_timeout_seconds": 1,
        "fact_check_forward_max_fetch": 3,
    }
    return plugin


class ForwardExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_forward_placeholder_is_expanded(self) -> None:
        plugin = make_plugin()
        event = FakeEvent([Reply(id="123456", message_str="[CQ:forward,id=abc123]")])
        FakeForwardClient.calls = []

        with (
            patch(
                "astrbot_plugin_fact_check.main.extract_quoted_message_text",
                new=AsyncMock(return_value="[CQ:forward,id=abc123]"),
            ),
            patch(
                "astrbot_plugin_fact_check.main.extract_quoted_message_images",
                new=AsyncMock(return_value=[]),
            ),
            patch("astrbot_plugin_fact_check.main.OneBotClient", FakeForwardClient),
            patch("astrbot_plugin_fact_check.main.OneBotPayloadParser", FakeForwardParser),
        ):
            request = await plugin._build_fact_check_request(event, trigger_text="/事实核查")

        self.assertEqual(FakeForwardClient.calls, ["abc123"])
        self.assertIn("岩浆垃圾桶", request.text)
        self.assertNotIn("[CQ:forward", request.text)

    async def test_direct_forward_component_is_expanded(self) -> None:
        plugin = make_plugin()
        event = FakeEvent([Forward(id="direct-forward-id")])
        FakeForwardClient.calls = []

        with (
            patch("astrbot_plugin_fact_check.main.OneBotClient", FakeForwardClient),
            patch("astrbot_plugin_fact_check.main.OneBotPayloadParser", FakeForwardParser),
        ):
            request = await plugin._build_fact_check_request(event, trigger_text="/事实核查")

        self.assertEqual(FakeForwardClient.calls, ["direct-forward-id"])
        self.assertIn("老破B", request.text)

    def test_multimsg_json_resid_is_treated_as_forward_id(self) -> None:
        plugin = make_plugin()
        text = (
            "[CQ:json,data={\"app\":\"com.tencent.multimsg\"&#44;"
            "\"config\":{\"forward\":1}&#44;"
            "\"meta\":{\"detail\":{\"resid\":\"resid-xyz\"}}}]"
        )

        self.assertEqual(plugin._extract_forward_ids_from_text(text), ["resid-xyz"])


if __name__ == "__main__":
    unittest.main()
