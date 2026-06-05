from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fact_check import (
    AnysearchEvidence,
    ClaimCandidate,
    FactCheckRequest,
    build_anysearch_queries,
    collect_anysearch_evidence,
    extract_public_urls,
    is_public_http_url,
    normalize_anysearch_query,
    run_fact_check,
)


class AnysearchEvidenceTests(unittest.TestCase):
    def test_normalize_anysearch_query_removes_fact_check_wrapping(self) -> None:
        self.assertEqual(
            normalize_anysearch_query("请核查：美国硅谷今天发生 6.0 级地震是否属实？"),
            "美国硅谷今天发生 6.0 级地震",
        )

    def test_extract_public_urls_dedupes_and_safety_filter_blocks_private_urls(self) -> None:
        text = """
        ### 1. Example
        - **URL**: https://example.com/a
        - **URL**: https://example.com/a
        - **URL**: http://127.0.0.1/admin
        Also see https://news.example.org/story).
        """
        urls = extract_public_urls(text)
        self.assertIn("https://example.com/a", urls)
        self.assertIn("http://127.0.0.1/admin", urls)
        self.assertIn("https://news.example.org/story", urls)
        self.assertFalse(is_public_http_url("http://127.0.0.1/admin"))
        self.assertFalse(is_public_http_url("http://10.0.0.5/status"))
        self.assertTrue(is_public_http_url("https://example.com/a"))

    def test_build_anysearch_queries_clamps_and_filters(self) -> None:
        queries = build_anysearch_queries(
            [
                ClaimCandidate("请核查：A 事件是否属实？", priority=5),
                ClaimCandidate("请核查：B 事件是否属实？", priority=4),
            ],
            max_claims=10,
            max_results_per_claim=99,
            freshness="week",
            content_types="web,news,invalid",
        )
        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0]["query"], "A 事件")
        self.assertEqual(queries[0]["max_results"], 10)
        self.assertEqual(queries[0]["freshness"], "week")
        self.assertEqual(queries[0]["content_types"], ["web", "news"])

    def test_collect_anysearch_evidence_uses_batch_search_and_extracts_public_pages(self) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_call_tool(*, tool_name, arguments, endpoint, api_key, timeout, max_retries=1):
            calls.append((tool_name, arguments))
            if tool_name == "batch_search":
                return (
                    "## Query 1\n"
                    "- **URL**: https://example.com/source-a\n"
                    "- Snippet A\n"
                    "## Query 2\n"
                    "- **URL**: http://127.0.0.1/private\n"
                    "- **URL**: https://example.org/source-b\n"
                )
            if tool_name == "extract":
                return f"## Extracted\n正文来自 {arguments['url']}"
            raise AssertionError(f"unexpected tool: {tool_name}")

        with patch("fact_check.anysearch_call_tool", side_effect=fake_call_tool):
            evidence = collect_anysearch_evidence(
                [
                    ClaimCandidate("请核查：A 事件是否属实？", priority=5),
                    ClaimCandidate("请核查：B 事件是否属实？", priority=4),
                ],
                enabled=True,
                endpoint="https://api.anysearch.com/mcp",
                api_key="",
                timeout=5,
                max_claims=3,
                max_results_per_claim=3,
                extract_top_urls=2,
                max_chars=4000,
            )

        self.assertIn("搜索摘要", evidence.text)
        self.assertIn("网页正文摘录", evidence.text)
        self.assertIn("https://example.com/source-a", evidence.text)
        self.assertIn("https://example.org/source-b", evidence.text)
        self.assertNotIn("正文来自 http://127.0.0.1/private", evidence.text)
        self.assertEqual(calls[0][0], "batch_search")
        self.assertEqual([call[0] for call in calls].count("extract"), 2)
        self.assertIn("ok; queries=2", evidence.reason)

    def test_collect_anysearch_evidence_disabled_does_not_call_network(self) -> None:
        with patch("fact_check.anysearch_call_tool") as mocked:
            evidence = collect_anysearch_evidence(
                [ClaimCandidate("请核查：A 事件是否属实？")],
                enabled=False,
                endpoint="https://api.anysearch.com/mcp",
                api_key="",
                timeout=5,
                max_claims=3,
                max_results_per_claim=3,
                extract_top_urls=2,
                max_chars=4000,
            )
        mocked.assert_not_called()
        self.assertEqual(evidence.text, "")
        self.assertEqual(evidence.sources, [])

    def test_run_fact_check_injects_anysearch_evidence_into_final_prompt(self) -> None:
        captured: dict[str, str] = {}

        def fake_generate_with_fallback(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return (
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "事实核查：大致可信\n要点：1. 有公开来源支持。"
                                    }
                                ]
                            }
                        }
                    ]
                },
                "gemini-test",
            )

        with (
            patch(
                "fact_check.extract_claims_from_text",
                return_value=[ClaimCandidate("请核查：A 事件是否属实？", "用户文字", 5)],
            ) as extract_claims,
            patch(
                "fact_check.collect_anysearch_evidence",
                return_value=AnysearchEvidence(
                    text="搜索摘要：\n- **URL**: https://example.com/source\n- A 事件报道",
                    sources=["https://example.com/source"],
                    reason="ok; queries=1 urls=1 extracts=0",
                ),
            ) as collect_evidence,
            patch("fact_check.generate_with_fallback", side_effect=fake_generate_with_fallback),
        ):
            result = run_fact_check(
                request_data=FactCheckRequest(text="A 事件是真的", trigger_text="/事实核查"),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-pre",
                main_models=["gemini-main"],
                anysearch_enabled=True,
            )

        extract_claims.assert_called_once()
        collect_evidence.assert_called_once()
        self.assertIn("Anysearch 预检索证据", captured["prompt"])
        self.assertIn("https://example.com/source", captured["prompt"])
        self.assertEqual(result.sources, ["https://example.com/source"])
        self.assertIn("事实核查：大致可信", result.reply)


if __name__ == "__main__":
    unittest.main()
