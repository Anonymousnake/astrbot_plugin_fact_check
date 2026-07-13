from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot.api.message_components import Image
from astrbot_plugin_fact_check import fact_check, main
from astrbot_plugin_fact_check.fact_check import FactCheckRequest, ImageInput


class LocalImage(Image):
    def __init__(self, source: Path) -> None:
        super().__init__(file=str(source))

    async def convert_to_file_path(self):
        return str(self.file)


class FactCheckResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_input_is_snapshotted_before_background_job(self) -> None:
        plugin = object.__new__(main.FactCheckPlugin)
        plugin.config = {"fact_check_image_download_timeout_seconds": 1}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "astrbot-temp.png"
            source.write_bytes(b"temporary-image")

            with patch.object(main.StarTools, "get_data_dir", return_value=root / "data"):
                images = await plugin._image_inputs([LocalImage(source)], remaining=1)

            self.assertEqual(len(images), 1)
            self.assertNotEqual(Path(images[0].path), source)
            self.assertEqual(Path(images[0].path).parent, root / "data" / "input_cache")
            source.unlink()
            self.assertEqual(Path(images[0].path).read_bytes(), b"temporary-image")

    def test_text_preprocess_failure_falls_through_to_main_check(self) -> None:
        response = {
            "candidates": [{"content": {"parts": [{"text": "事实核查：证据不足"}]}}],
        }

        with (
            patch.object(fact_check, "extract_claims_from_text", side_effect=RuntimeError("preprocess failed")),
            patch.object(fact_check, "generate_with_fallback", return_value=(response, "gemini-2.5-flash")),
        ):
            result = fact_check.run_fact_check(
                request_data=FactCheckRequest(text="某条需要核查的消息", trigger_text="/事实核查"),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                main_models=["gemini-2.5-flash"],
            )

        self.assertEqual(result.reply, "事实核查：证据不足")
        self.assertTrue(result.reason.startswith("ok"))

    def test_image_preprocess_failure_still_sends_image_to_main_check(self) -> None:
        response = {
            "candidates": [{"content": {"parts": [{"text": "事实核查：证据不足"}]}}],
        }
        attached = [{"inline_data": {"mime_type": "image/png", "data": "AA=="}}]

        with (
            patch.object(fact_check, "extract_claims_from_images", side_effect=RuntimeError("image parse failed")),
            patch.object(fact_check, "build_inline_image_parts", return_value=attached),
            patch.object(fact_check, "generate_with_fallback", return_value=(response, "gemini-2.5-flash")) as generate,
        ):
            result = fact_check.run_fact_check(
                request_data=FactCheckRequest(
                    text="",
                    trigger_text="/事实核查",
                    images=[ImageInput(url="", path="missing-after-staging.png")],
                ),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                main_models=["gemini-2.5-flash"],
            )

        self.assertEqual(result.reply, "事实核查：证据不足")
        self.assertEqual(generate.call_args.kwargs["extra_parts"], attached)

    def test_best_model_can_use_anysearch_without_google_grounding(self) -> None:
        response = {
            "candidates": [{"content": {"parts": [{"text": "事实核查：可信"}]}}],
        }

        with patch.object(fact_check, "gemini_generate", return_value=response) as generate:
            body, model = fact_check.generate_with_fallback(
                prompt="Use the supplied evidence.",
                models=["gemini-3.5-flash", "gemini-2.5-flash"],
                api_key="test-key",
                base_url="https://example.invalid/models",
                temperature=0,
                max_output_tokens=128,
                grounding=True,
                ungrounded_models=["gemini-3.5-flash"],
            )

        self.assertEqual(body, response)
        self.assertEqual(model, "gemini-3.5-flash")
        self.assertFalse(generate.call_args.kwargs["grounding"])

    def test_unavailable_best_model_is_skipped_during_cooldown(self) -> None:
        request = fact_check.httpx.Request("POST", "https://example.invalid/models/generateContent")
        response = fact_check.httpx.Response(503, request=request)
        unavailable = fact_check.httpx.HTTPStatusError("busy", request=request, response=response)
        success = {"candidates": [{"content": {"parts": [{"text": "事实核查：可信"}]}}]}

        with (
            patch.object(fact_check, "_MODEL_FAILURE_UNTIL", {}),
            patch.object(fact_check, "gemini_generate", side_effect=[unavailable, success, success]) as generate,
            patch.object(fact_check.time, "sleep"),
        ):
            first_body, first_model = fact_check.generate_with_fallback(
                prompt="Use the supplied evidence.",
                models=["gemini-3.5-flash", "gemini-2.5-flash"],
                api_key="test-key",
                base_url="https://example.invalid/models",
                temperature=0,
                max_output_tokens=128,
                grounding=True,
                ungrounded_models=["gemini-3.5-flash"],
                model_failure_cooldown_seconds=900,
            )
            second_body, second_model = fact_check.generate_with_fallback(
                prompt="Use the supplied evidence.",
                models=["gemini-3.5-flash", "gemini-2.5-flash"],
                api_key="test-key",
                base_url="https://example.invalid/models",
                temperature=0,
                max_output_tokens=128,
                grounding=True,
                ungrounded_models=["gemini-3.5-flash"],
                model_failure_cooldown_seconds=900,
            )

        self.assertEqual(first_body, success)
        self.assertEqual(second_body, success)
        self.assertEqual(first_model, "gemini-2.5-flash")
        self.assertEqual(second_model, "gemini-2.5-flash")
        self.assertEqual([call.kwargs["model"] for call in generate.call_args_list], [
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-flash",
        ])


if __name__ == "__main__":
    unittest.main()
