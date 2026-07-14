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
    def test_gemini_3_uses_default_temperature_and_explicit_thinking_level(self) -> None:
        with patch.object(fact_check, "post_json_with_timeout", return_value={}) as post:
            fact_check.gemini_generate(
                prompt="Review evidence.",
                model="gemini-3-flash-preview",
                api_key="test-key",
                base_url="https://example.invalid/models",
                temperature=0.1,
                max_output_tokens=512,
                grounding=False,
                thinking_level="medium",
            )

        config = post.call_args.args[1]["generationConfig"]
        self.assertNotIn("temperature", config)
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "medium"})

    def test_gemini_25_keeps_low_temperature_and_disables_thinking(self) -> None:
        config = fact_check.build_generation_config(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_output_tokens=512,
        )

        self.assertEqual(config["temperature"], 0.1)
        self.assertEqual(config["thinkingConfig"], {"thinkingBudget": 0})

    def test_grounding_support_mapping_and_anysearch_excerpts_reach_verdict(self) -> None:
        evidence_response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "The policy exists."}]},
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/policy", "title": "Policy"}},
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"text": "The policy exists."},
                                "groundingChunkIndices": [0],
                            },
                        ],
                    },
                },
            ],
        }
        verdict_response = {
            "candidates": [{"content": {"parts": [{"text": "事实核查：部分存疑"}]}}],
        }

        with (
            patch.object(
                fact_check,
                "extract_claims_from_text",
                return_value=[fact_check.ClaimCandidate("Check the policy implication.")],
            ),
            patch.object(
                fact_check,
                "collect_anysearch_evidence",
                return_value=fact_check.AnysearchEvidence(
                    text="网页正文摘录：具体商品适用范围未明确。",
                    sources=["https://example.org/report"],
                ),
            ),
            patch.object(
                fact_check,
                "generate_with_fallback",
                side_effect=[
                    (evidence_response, "gemini-2.5-flash"),
                    (verdict_response, "gemini-3-flash-preview"),
                ],
            ) as generate,
        ):
            fact_check.run_fact_check(
                request_data=FactCheckRequest(text="A policy claim", trigger_text="/factcheck"),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                evidence_model="gemini-2.5-flash",
                verdict_models=["gemini-3-flash-preview"],
                anysearch_enabled=True,
            )

        verdict_prompt = generate.call_args_list[1].kwargs["prompt"]
        self.assertIn("证据片段：The policy exists.", verdict_prompt)
        self.assertIn("https://example.com/policy", verdict_prompt)
        self.assertIn("具体商品适用范围未明确", verdict_prompt)
        self.assertEqual(generate.call_args_list[1].kwargs["thinking_level"], "medium")

    def test_text_with_image_uses_one_multimodal_preprocess_and_reuses_inline_parts(self) -> None:
        inline_parts = [{"inline_data": {"mime_type": "image/png", "data": "AA=="}}]
        response = {"candidates": [{"content": {"parts": [{"text": "事实核查：可信"}]}}]}

        with (
            patch.object(fact_check, "build_inline_image_parts", return_value=inline_parts) as build_images,
            patch.object(
                fact_check,
                "extract_claims_from_images",
                return_value=[fact_check.ClaimCandidate("Check image and caption.")],
            ) as extract_images,
            patch.object(fact_check, "extract_claims_from_text") as extract_text,
            patch.object(fact_check, "generate_with_fallback", return_value=(response, "gemini-2.5-flash")) as generate,
        ):
            fact_check.run_fact_check(
                request_data=FactCheckRequest(
                    text="caption",
                    trigger_text="/factcheck",
                    images=[ImageInput(url="https://example.com/image.png")],
                ),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                main_models=["gemini-2.5-flash"],
            )

        build_images.assert_called_once()
        extract_text.assert_not_called()
        self.assertIs(extract_images.call_args.kwargs["inline_parts"], inline_parts)
        self.assertIs(generate.call_args.kwargs["extra_parts"], inline_parts)

    def test_text_preprocess_is_used_as_fallback_when_multimodal_extracts_no_claims(self) -> None:
        inline_parts = [{"inline_data": {"mime_type": "image/png", "data": "AA=="}}]
        response = {"candidates": [{"content": {"parts": [{"text": "事实核查：可信"}]}}]}

        with (
            patch.object(fact_check, "build_inline_image_parts", return_value=inline_parts),
            patch.object(fact_check, "extract_claims_from_images", return_value=[]),
            patch.object(
                fact_check,
                "extract_claims_from_text",
                return_value=[fact_check.ClaimCandidate("Check caption fallback.")],
            ) as extract_text,
            patch.object(fact_check, "generate_with_fallback", return_value=(response, "gemini-2.5-flash")),
        ):
            result = fact_check.run_fact_check(
                request_data=FactCheckRequest(
                    text="caption",
                    trigger_text="/factcheck",
                    images=[ImageInput(url="https://example.com/image.png")],
                ),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                main_models=["gemini-2.5-flash"],
            )

        extract_text.assert_called_once()
        self.assertEqual(result.candidates[0].claim, "Check caption fallback.")

    def test_total_deadline_bounds_each_http_timeout_and_resets_after_call(self) -> None:
        observed: list[float] = []

        @fact_check._with_request_deadline
        def sample(*, total_timeout_seconds: int) -> None:
            observed.append(fact_check._bounded_timeout(30))

        sample(total_timeout_seconds=1)

        self.assertEqual(len(observed), 1)
        self.assertGreater(observed[0], 0)
        self.assertLessEqual(observed[0], 1)
        self.assertIsNone(fact_check._REQUEST_DEADLINE.get())

    def test_expired_total_deadline_prevents_a_new_http_attempt(self) -> None:
        token = fact_check._REQUEST_DEADLINE.set(fact_check.time.monotonic() - 1)
        try:
            with patch.object(fact_check.httpx, "Client") as client:
                with self.assertRaises(fact_check.httpx.TimeoutException):
                    fact_check.post_json_with_timeout(
                        "https://example.invalid",
                        {},
                        api_key="test-key",
                        timeout=30,
                    )
        finally:
            fact_check._REQUEST_DEADLINE.reset(token)

        client.assert_not_called()

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

    async def test_equal_image_bytes_use_a_stable_snapshot_and_digest(self) -> None:
        plugin = object.__new__(main.FactCheckPlugin)
        plugin.config = {"fact_check_image_download_timeout_seconds": 1}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same-image")
            second.write_bytes(b"same-image")

            with patch.object(main.StarTools, "get_data_dir", return_value=root / "data"):
                first_input = await plugin._image_inputs([LocalImage(first)], remaining=1)
                second_input = await plugin._image_inputs([LocalImage(second)], remaining=1)

            self.assertEqual(first_input[0].path, second_input[0].path)
            self.assertEqual(first_input[0].content_sha256, second_input[0].content_sha256)

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

    def test_grounded_evidence_is_reviewed_by_ungrounded_gemini_3(self) -> None:
        evidence_response = {
            "candidates": [{"content": {"parts": [{"text": "Grounded evidence: the policy exists."}]}}],
        }
        verdict_response = {
            "candidates": [{"content": {"parts": [{"text": "Fact check: partial support."}]}}],
        }

        with (
            patch.object(
                fact_check,
                "extract_claims_from_text",
                return_value=[fact_check.ClaimCandidate("Check the policy and its product implication.")],
            ),
            patch.object(
                fact_check,
                "generate_with_fallback",
                side_effect=[
                    (evidence_response, "gemini-2.5-flash"),
                    (verdict_response, "gemini-3-flash-preview"),
                ],
            ) as generate,
        ):
            result = fact_check.run_fact_check(
                request_data=FactCheckRequest(text="A policy claim", trigger_text="/factcheck"),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                evidence_model="gemini-2.5-flash",
                verdict_models=["gemini-3-flash-preview"],
            )

        evidence_call, verdict_call = generate.call_args_list
        self.assertEqual(evidence_call.kwargs["models"], ["gemini-2.5-flash"])
        self.assertTrue(evidence_call.kwargs["grounding"])
        self.assertEqual(verdict_call.kwargs["models"], ["gemini-3-flash-preview"])
        self.assertFalse(verdict_call.kwargs["grounding"])
        self.assertIn("Grounded evidence", verdict_call.kwargs["prompt"])
        self.assertEqual(result.reply, "Fact check: partial support.")

    def test_grounded_evidence_is_the_complete_fallback_when_gemini_3_fails(self) -> None:
        evidence_response = {
            "candidates": [{"content": {"parts": [{"text": "Fact check: evidence-model fallback."}]}}],
        }

        with (
            patch.object(
                fact_check,
                "extract_claims_from_text",
                return_value=[fact_check.ClaimCandidate("Check the claim.")],
            ),
            patch.object(
                fact_check,
                "generate_with_fallback",
                side_effect=[
                    (evidence_response, "gemini-2.5-flash"),
                    RuntimeError("Gemini 3 unavailable"),
                ],
            ),
        ):
            result = fact_check.run_fact_check(
                request_data=FactCheckRequest(text="A claim", trigger_text="/factcheck"),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                evidence_model="gemini-2.5-flash",
                verdict_models=["gemini-3-flash-preview"],
            )

        self.assertEqual(result.reply, "Fact check: evidence-model fallback.")

    def test_grounded_evidence_is_used_when_gemini_3_returns_no_text(self) -> None:
        evidence_response = {
            "candidates": [{"content": {"parts": [{"text": "Fact check: grounded fallback."}]}}],
        }
        empty_verdict = {"candidates": [{"content": {"parts": []}}]}

        with (
            patch.object(
                fact_check,
                "extract_claims_from_text",
                return_value=[fact_check.ClaimCandidate("Check the claim.")],
            ),
            patch.object(
                fact_check,
                "generate_with_fallback",
                side_effect=[
                    (evidence_response, "gemini-2.5-flash"),
                    (empty_verdict, "gemini-3-flash-preview"),
                ],
            ),
        ):
            result = fact_check.run_fact_check(
                request_data=FactCheckRequest(text="A claim", trigger_text="/factcheck"),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-3.1-flash-lite",
                evidence_model="gemini-2.5-flash",
                verdict_models=["gemini-3-flash-preview"],
            )

        self.assertEqual(result.reply, "Fact check: grounded fallback.")

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
