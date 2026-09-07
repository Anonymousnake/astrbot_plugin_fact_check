from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from astrbot_plugin_fact_check import fact_check as fc
from astrbot_plugin_fact_check import main, serpapi_search


class SerpApiTransportTests(unittest.TestCase):
    def test_search_returns_bounded_organic_results_and_freshness(self):
        response = MagicMock(status=200)
        response.read1.side_effect = [
            json.dumps(
                {
                    "organic_results": [
                        {
                            "title": "公告",
                            "link": "https://agency.gov/report",
                            "snippet": "公告正文摘要",
                        },
                        {
                            "title": "第二条",
                            "link": "https://other.example/report",
                            "snippet": "另一条摘要",
                        },
                    ]
                }
            ).encode(),
            b"",
        ]
        with patch.object(serpapi_search.http.client, "HTTPSConnection") as transport:
            transport.return_value.getresponse.return_value = response
            hits = serpapi_search.search(
                "公开公告",
                api_key="test-secret",
                timeout=5,
                max_results=1,
                freshness="week",
            )
        self.assertEqual(
            hits,
            [
                {
                    "title": "公告",
                    "url": "https://agency.gov/report",
                    "snippet": "公告正文摘要",
                }
            ],
        )
        request = transport.return_value.request.call_args
        self.assertEqual(request.args[0], "GET")
        self.assertIn("tbs=qdr%3Aw", request.args[1])
        transport.return_value.close.assert_called_once()

    def test_provider_errors_do_not_echo_credentials(self):
        for status, body in (
            (401, {"error": "invalid key test-secret"}),
            (200, {"error": "request failed for api_key=test-secret"}),
            (200, {"organic_results": {"unexpected": "test-secret"}}),
        ):
            with self.subTest(status=status):
                response = MagicMock(status=status)
                response.read1.side_effect = [json.dumps(body).encode(), b""]
                with patch.object(
                    serpapi_search.http.client, "HTTPSConnection"
                ) as transport:
                    transport.return_value.getresponse.return_value = response
                    with self.assertRaises(RuntimeError) as raised:
                        serpapi_search.search("query", api_key="test-secret", timeout=5)
                self.assertNotIn("test-secret", str(raised.exception))

    def test_response_size_limit_closes_the_connection(self):
        response = MagicMock(status=200)
        response.read1.return_value = b"x" * 1_000_001
        with patch.object(serpapi_search.http.client, "HTTPSConnection") as transport:
            transport.return_value.getresponse.return_value = response
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                serpapi_search.search("query", api_key="test-key", timeout=5)
        transport.return_value.close.assert_called_once()

    def test_elapsed_budget_does_not_start_reading_a_response(self):
        with (
            patch.object(serpapi_search.http.client, "HTTPSConnection") as transport,
            patch.object(serpapi_search.time, "monotonic", side_effect=[10.0, 12.0]),
            self.assertRaisesRegex(RuntimeError, "TimeoutError"),
        ):
            serpapi_search.search("query", api_key="test-key", timeout=1)
        transport.return_value.getresponse.assert_not_called()
        transport.return_value.close.assert_called_once()

    def test_malformed_results_and_unsafe_links_are_not_evidence(self):
        response = MagicMock(status=200)
        response.read1.side_effect = [
            json.dumps(
                {
                    "organic_results": [
                        None,
                        {"link": "file:///etc/passwd"},
                        {"link": "https://user:password@example.org/a"},
                    ]
                }
            ).encode(),
            b"",
        ]
        with patch.object(serpapi_search.http.client, "HTTPSConnection") as transport:
            transport.return_value.getresponse.return_value = response
            self.assertEqual(
                serpapi_search.search("query", api_key="test-key", timeout=5), []
            )


class SearchFallbackTests(unittest.TestCase):
    def test_pipeline_uses_backup_when_anysearch_fails(self):
        claim = fc.ClaimCandidate("甲公司发布新手机。")
        body = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "事实核查：可信\n1. 核查点：甲公司发布新手机。\n结论：已核实\n"
                                    "依据：公司公告列明了新手机的发布信息。\n证据关系：支持一致"
                                )
                            }
                        ]
                    },
                }
            ]
        }
        with (
            patch.object(fc, "extract_claims_from_text", return_value=[claim]),
            patch.object(
                fc,
                "collect_anysearch_evidence",
                return_value=fc.AnysearchEvidence(status="search_failed"),
            ),
            patch.object(
                fc,
                "search_serpapi",
                return_value=[
                    {
                        "title": "公告",
                        "url": "https://agency.gov/phone",
                        "snippet": claim.claim,
                    }
                ],
            ),
            patch.object(fc, "fetch_public_page_text", return_value=claim.claim),
            patch.object(fc, "ensure_public_url_target"),
            patch.object(
                fc, "generate_with_fallback", return_value=(body, "test-model")
            ) as generate,
        ):
            result = fc.run_fact_check(
                request_data=fc.FactCheckRequest(
                    text=claim.claim, trigger_text="/事实核查"
                ),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="test-model",
                anysearch_enabled=True,
                serpapi_enabled=True,
                serpapi_api_key="test-search-key",
            )
        self.assertIn("SerpAPI 预检索证据", generate.call_args.kwargs["prompt"])
        self.assertIn("结论：已核实", result.reply)
        self.assertEqual(result.sources, ["https://agency.gov/phone"])

    def test_backup_settings_invalidate_the_reply_cache(self):
        plugin = object.__new__(main.FactCheckPlugin)
        request = fc.FactCheckRequest(text="测试命题", trigger_text="/事实核查")
        plugin.config = {}
        original = plugin._request_cache_key(request)
        for name, value in (
            ("fact_check_serpapi_enabled", True),
            ("fact_check_serpapi_api_key", "test-key"),
            ("fact_check_serpapi_timeout_seconds", 12),
            ("fact_check_serpapi_max_queries", 1),
        ):
            with self.subTest(setting=name):
                plugin.config = {name: value}
                self.assertNotEqual(plugin._request_cache_key(request), original)

    def test_backup_query_cap_and_private_url_rejection(self):
        claims = [fc.ClaimCandidate(f"公司{index}发布手机。") for index in range(3)]
        primary = fc.AnysearchEvidence(status="search_failed")
        with (
            patch.object(
                fc,
                "search_serpapi",
                return_value=[
                    {
                        "title": "内部",
                        "url": "http://127.0.0.1/private",
                        "snippet": "不应抓取",
                    }
                ],
            ) as search,
            patch.object(fc, "fetch_public_page_text") as fetch,
        ):
            result = fc.collect_serpapi_fallback(
                primary, claims, api_key="test-key", max_queries=2
            )
        self.assertEqual(search.call_count, 2)
        fetch.assert_not_called()
        self.assertIs(result, primary)

    def test_healthy_primary_does_not_spend_backup_queries(self):
        primary = fc.AnysearchEvidence(
            text="原证据", status="ok", claim_sources=[["https://agency.gov/report"]]
        )
        with patch.object(fc, "search_serpapi") as search:
            result = fc.collect_serpapi_fallback(
                primary, [fc.ClaimCandidate("某政策已经发布")], api_key="test-key"
            )
        self.assertIs(result, primary)
        search.assert_not_called()

    def test_fallback_only_searches_missing_claim_and_keeps_existing_sources(self):
        claims = [
            fc.ClaimCandidate("甲公司发布新手机。"),
            fc.ClaimCandidate("乙公司发布新电脑。"),
        ]
        primary = fc.AnysearchEvidence(
            text="甲公司证据",
            status="partial",
            sources=["https://a.example/phone"],
            claim_sources=[["https://a.example/phone"], []],
        )
        with (
            patch.object(
                fc,
                "search_serpapi",
                return_value=[
                    {
                        "title": "乙公司公告",
                        "url": "https://b.example/computer",
                        "snippet": "乙公司发布新电脑。",
                    }
                ],
            ) as search,
            patch.object(
                fc,
                "fetch_public_page_text",
                return_value="乙公司发布新电脑。公告列明了产品信息。",
            ),
        ):
            result = fc.collect_serpapi_fallback(primary, claims, api_key="test-key")
        self.assertEqual(search.call_count, 1)
        self.assertIn("乙公司", search.call_args.args[0])
        self.assertEqual(
            result.claim_sources,
            [["https://a.example/phone"], ["https://b.example/computer"]],
        )
        self.assertIn("甲公司证据", result.text)
        self.assertIn("乙公司发布新电脑", result.text)
        self.assertEqual(result.provider, "Anysearch + SerpAPI")
        self.assertEqual(primary.claim_sources, [["https://a.example/phone"], []])

    def test_search_snippets_are_not_promoted_to_direct_evidence(self):
        with (
            patch.object(
                fc,
                "search_serpapi",
                return_value=[
                    {
                        "title": "公告",
                        "url": "https://agency.gov/report",
                        "snippet": "该药物已获批准。",
                    }
                ],
            ),
            patch.object(
                fc, "fetch_public_page_text", side_effect=RuntimeError("unavailable")
            ),
        ):
            result = fc.collect_serpapi_fallback(
                fc.AnysearchEvidence(status="search_failed"),
                [fc.ClaimCandidate("该药物已获批准。")],
                api_key="test-key",
            )
        self.assertIn("该药物已获批准", result.text)
        self.assertEqual(result.claim_sources, [[]])
        self.assertEqual(result.status, "search_only")

    def test_exhausted_budget_skips_backup(self):
        primary = fc.AnysearchEvidence(status="search_failed")
        with (
            patch.object(fc, "_retry_budget_available", return_value=False),
            patch.object(fc, "search_serpapi") as search,
        ):
            self.assertIs(
                fc.collect_serpapi_fallback(
                    primary, [fc.ClaimCandidate("事实命题")], api_key="test-key"
                ),
                primary,
            )
        search.assert_not_called()

    def test_backup_outage_keeps_primary_evidence(self):
        primary = fc.AnysearchEvidence(text="已有摘要", status="search_only")
        with patch.object(
            fc, "search_serpapi", side_effect=RuntimeError("unavailable")
        ):
            result = fc.collect_serpapi_fallback(
                primary, [fc.ClaimCandidate("事实命题")], api_key="test-key"
            )
        self.assertIs(result, primary)


if __name__ == "__main__":
    unittest.main()
