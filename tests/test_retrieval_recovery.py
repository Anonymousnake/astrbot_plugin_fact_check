from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_fact_check import fact_check as fc


class RetrievalRecoveryTests(unittest.TestCase):
    def test_government_country_sources_survive_link_selection(self):
        for host in ("www.chp.gov.hk", "www.health.gov.au", "www.gov.uk"):
            with self.subTest(host=host):
                source = f"https://{host}/resources/report"
                self.assertEqual(fc.select_fact_check_sources([source], []), [source])
        for host in ("who.int.evil.example", "agency.gov.hk.evil.example"):
            self.assertEqual(fc.select_fact_check_sources([f"https://{host}/report"], []), [])

    def test_current_state_and_calendar_dates_do_not_exclude_old_sources(self):
        for claim in (
            "目前这个法律仍然有效", "当前疾病的治疗方法", "去年发布的公告",
            "下月举行的大会", "即将开放的博物馆", "The known result is correct",
        ):
            with self.subTest(claim=claim):
                self.assertEqual(fc.infer_anysearch_freshness(claim), "")
        self.assertEqual(fc.infer_anysearch_freshness("今天发生的事故"), "day")

    def test_summary_only_mismatch_is_repaired_without_regenerating_evidence(self):
        claim = "Alpha event occurred."
        body = {"candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"text": (
                f"事实核查：基本不实\n1. 核查点：{claim}\n结论：已核实\n"
                "依据：官方公告明确确认了该事件。\n证据关系：支持一致"
            )}]},
            "groundingMetadata": {
                "groundingChunks": [{"web": {"title": "Agency", "uri": "https://agency.gov/report"}}],
                "groundingSupports": [{"segment": {"text": claim}, "groundingChunkIndices": [0]}],
            },
        }]}
        original = copy.deepcopy(body)
        with (
            patch.object(fc, "build_inline_image_parts", return_value=[]),
            patch.object(fc, "extract_claims_from_text", return_value=[fc.ClaimCandidate(claim)]),
            patch.object(fc, "generate_with_fallback", return_value=(body, "test")) as generate,
        ):
            result = fc.run_fact_check(
                request_data=fc.FactCheckRequest(claim, "/factcheck"),
                api_key="test", base_url="https://example.invalid", pre_model="test",
                evidence_model="test", verdict_models=[],
            )
        generate.assert_called_once()
        self.assertEqual(body, original, "Grounding byte offsets must keep their original text")
        self.assertIn("事实核查：可信", result.reply)
        self.assertFalse(result.reason.startswith("ok; partial"))
        self.assertTrue(result.sources)

    def test_failed_or_irrelevant_first_page_tries_next_search_result(self):
        for first in ("failed", "irrelevant", "useful"):
            with self.subTest(first=first):
                extracted = []

                def call(*, tool_name, arguments, **kwargs):
                    if tool_name == "search":
                        return "\n".join(f"https://example.com/{i}" for i in range(4))
                    url = arguments["url"]
                    extracted.append(url)
                    if url.endswith("/0"):
                        if first == "failed":
                            raise RuntimeError("page unavailable")
                        if first == "irrelevant":
                            return "Unrelated weather forecast."
                    return "Alpha event occurred according to the official report."

                with (
                    patch.object(fc, "anysearch_call_tool", side_effect=call),
                    patch.object(fc, "ensure_public_url_target"),
                ):
                    evidence = fc.collect_anysearch_evidence(
                        [fc.ClaimCandidate("Alpha event occurred.")], enabled=True,
                        endpoint="https://example.com/mcp", api_key="", timeout=5,
                        anysearch_direct_fetch_fallback=False, max_claims=1,
                        max_results_per_claim=4, extract_top_urls=1, max_chars=4000,
                    )
                expected = "0" if first == "useful" else "1"
                self.assertIn(f"https://example.com/{expected}", evidence.claim_sources[0])
                self.assertEqual(len(extracted), 1 if first == "useful" else 2)

    def test_summary_repair_still_rejects_wrong_evidence_direction(self):
        body = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": (
            "事实核查：基本不实\n1. 核查点：Alpha event occurred.\n结论：已核实\n"
            "依据：报道证实该事件。\n证据关系：反驳一致"
        )}]}}]}
        with self.assertRaises(fc.IncompleteGenerationError):
            fc.validate_complete_fact_check_result(body, reconcile_summary=True)

    def test_recovery_caps_requests_and_keeps_unrelated_pages_out_of_evidence(self):
        def call(*, tool_name, **kwargs):
            if tool_name == "search":
                return "\n".join(f"https://example.com/{i}" for i in range(8))
            return "Unrelated weather forecast."

        for budget, expected_attempts in ((True, 3), (False, 1)):
            with (
                self.subTest(budget=budget),
                patch.object(fc, "anysearch_call_tool", side_effect=call),
                patch.object(fc, "ensure_public_url_target"),
            ):
                token = fc._REQUEST_DEADLINE.set(None if budget else fc.time.monotonic() - 1)
                try:
                    evidence = fc.collect_anysearch_evidence(
                        [fc.ClaimCandidate("Alpha event occurred.")], enabled=True,
                        endpoint="https://example.com/mcp", api_key="", timeout=5,
                        anysearch_direct_fetch_fallback=False, max_claims=1,
                        max_results_per_claim=8, extract_top_urls=1, max_chars=4000,
                    )
                finally:
                    fc._REQUEST_DEADLINE.reset(token)
                self.assertEqual(evidence.extract_attempted, expected_attempts)
                self.assertEqual(evidence.claim_sources, [[]])
