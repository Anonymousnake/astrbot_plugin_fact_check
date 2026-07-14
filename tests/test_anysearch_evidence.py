from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for astrbot_root in (Path("D:/Codex/AstrBot"), Path("/home/ubuntu/AstrBot")):
    if astrbot_root.exists():
        sys.path.insert(0, str(astrbot_root))

from fact_check import (
    AnysearchEvidence,
    ClaimCandidate,
    FactCheckRequest,
    ImageInput,
    append_source_links,
    build_anysearch_queries,
    compact_source_label,
    collect_anysearch_evidence,
    dedupe_candidates,
    extract_public_urls,
    extract_claims_from_text,
    is_public_http_url,
    normalize_anysearch_query,
    run_fact_check,
    read_image_input_bytes,
    sanitize_anysearch_evidence_text,
    sanitize_fact_check_reply,
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

    def test_compact_source_label_prefers_domain_and_short_path(self) -> None:
        self.assertEqual(
            compact_source_label("https://www.iana.org/domains/reserved/example/path"),
            "iana.org/domains/reserved",
        )
        self.assertEqual(
            compact_source_label("Example Title：https://example.com/a/b/c"),
            "example.com/a/b",
        )

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
        self.assertIn("可核验链接：", result.reply)
        self.assertIn("https://example.com/source", result.reply)

    def test_final_reply_always_appends_clickable_source_urls(self) -> None:
        reply = append_source_links(
            "事实核查：可信\n来源：Example News",
            ["Example News：https://example.com/report"],
        )

        self.assertIn("可核验链接：", reply)
        self.assertIn("https://example.com/report", reply)

    def test_extraction_prompt_splits_high_risk_composite_claims(self) -> None:
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
                                        "text": """
                                        [
                                          {"question":"请核查：《人工智能拟人化互动服务管理暂行办法》是否存在及其生效时间是否属实？","source":"用户文字","priority":5},
                                          {"question":"请核查：该办法是否明确禁止亲密陪伴 AI 与性互硬件销售？","source":"用户文字","priority":5}
                                        ]
                                        """
                                    }
                                ]
                            }
                        }
                    ]
                },
                "gemini-test",
            )

        with patch("fact_check.generate_with_fallback", side_effect=fake_generate_with_fallback):
            claims = extract_claims_from_text(
                "网传《人工智能拟人化互动服务管理暂行办法》将实施，亲密陪伴 AI 和性互硬件以后不能卖。",
                model="gemini-pre",
                api_key="test-key",
                base_url="https://example.invalid/models",
            )

        self.assertIn("高风险复合命题要拆成 atomic claims", captured["prompt"])
        self.assertIn("至少拆成两个问题", captured["prompt"])
        self.assertGreaterEqual(len(claims), 2)
        self.assertTrue(any("是否存在" in claim.claim for claim in claims))
        self.assertTrue(any("是否明确禁止" in claim.claim for claim in claims))

    def test_main_prompt_calibrates_partial_support_for_regulation_claims(self) -> None:
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
                                        "text": (
                                            "事实核查：部分存疑\n"
                                            "要点：1. 已证实：法规存在。"
                                            "2. 未直接证实：未找到直接证据支持该具体推论。"
                                        )
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
                return_value=[
                    ClaimCandidate("请核查：《人工智能拟人化互动服务管理暂行办法》是否存在及其生效时间是否属实？", "用户文字", 5),
                    ClaimCandidate("请核查：该办法是否明确禁止亲密陪伴 AI 与性互硬件销售？", "用户文字", 5),
                ],
            ),
            patch("fact_check.generate_with_fallback", side_effect=fake_generate_with_fallback),
        ):
            result = run_fact_check(
                request_data=FactCheckRequest(
                    text="网传《人工智能拟人化互动服务管理暂行办法》将实施，亲密陪伴 AI 和性互硬件以后不能卖。",
                    trigger_text="/事实核查",
                ),
                api_key="test-key",
                base_url="https://example.invalid/models",
                pre_model="gemini-pre",
                main_models=["gemini-main"],
            )

        self.assertIn("整体结论最高为“部分存疑”", captured["prompt"])
        self.assertIn("不要因为任一子事实成立就把整体判成“可信”或“基本可信但需限定”", captured["prompt"])
        self.assertIn("已核实 / 条件性成立 / 表述需限定", captured["prompt"])
        self.assertNotIn("已证实 / 未直接证实 / 存疑", captured["prompt"])
        self.assertIn("事实核查：部分存疑", result.reply)

    def test_run_fact_check_continues_when_anysearch_search_fails(self) -> None:
        captured: dict[str, str] = {}

        def fake_generate_with_fallback(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return (
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "事实核查：暂无法确认\n要点：搜索失败但主核查继续。"}]
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
            ),
            patch("fact_check.anysearch_call_tool", side_effect=TimeoutError("simulated timeout")),
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

        self.assertIn("事实核查：暂无法确认", result.reply)
        self.assertEqual(result.sources, [])
        self.assertNotIn("搜索摘要：", captured["prompt"])
        self.assertNotIn("simulated timeout", captured["prompt"])

    def test_request_cache_key_changes_when_anysearch_retrieval_config_changes(self) -> None:
        from astrbot_plugin_fact_check.main import FactCheckPlugin

        request = FactCheckRequest(text="A 事件是真的", trigger_text="/事实核查")
        plugin = FactCheckPlugin.__new__(FactCheckPlugin)
        plugin.config = {
            "fact_check_anysearch_enabled": True,
            "fact_check_anysearch_endpoint": "https://api.anysearch.com/mcp",
            "fact_check_anysearch_timeout_seconds": 20,
            "fact_check_anysearch_max_claims": 3,
            "fact_check_anysearch_max_results_per_claim": 3,
            "fact_check_anysearch_extract_top_urls": 2,
            "fact_check_anysearch_max_chars": 6000,
            "fact_check_anysearch_freshness": "",
            "fact_check_anysearch_content_types": ["web", "news"],
        }
        original_key = plugin._request_cache_key(request)

        plugin.config = dict(plugin.config)
        plugin.config["fact_check_anysearch_content_types"] = ["news"]
        self.assertNotEqual(original_key, plugin._request_cache_key(request))

        plugin.config = dict(plugin.config)
        plugin.config["fact_check_anysearch_content_types"] = ["web", "news"]
        plugin.config["fact_check_anysearch_extract_top_urls"] = 0
        self.assertNotEqual(original_key, plugin._request_cache_key(request))

    def test_sanitize_anysearch_evidence_removes_markdown_url_labels(self) -> None:
        text = "### Query\n- **URL**: https://example.com/a\n- **Title**: Example"

        cleaned = sanitize_anysearch_evidence_text(text)

        self.assertNotIn("**", cleaned)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("- URL", cleaned)
        self.assertIn("URL：https://example.com/a", cleaned)

    def test_sanitize_fact_check_reply_splits_single_line_points(self) -> None:
        reply = "事实核查：**大致可信**\n要点：1. A 有依据。2. B 证据不足。\n来源：Example"

        cleaned = sanitize_fact_check_reply(reply)

        self.assertNotIn("**", cleaned)
        self.assertNotIn("要点：1.", cleaned)
        self.assertIn("1. 核查点：A 有依据。", cleaned)
        self.assertIn("\n2. 核查点：B 证据不足。", cleaned)

    def test_sanitize_fact_check_reply_preserves_conditional_verdict(self) -> None:
        reply = (
            "事实核查：已证实，但需满足条件\n"
            "1. 核查点：欧盟规则下私人飞机可以被视为符合绿色投资条件。\n"
            "结论：已证实，因为有条件规则\n"
            "依据：只有在满足特定排放和技术筛选条件时才适用。"
        )

        cleaned = sanitize_fact_check_reply(reply)

        self.assertNotIn("事实核查：已证实", cleaned)
        self.assertNotIn("结论：已证实", cleaned)
        self.assertIn("事实核查：条件性成立", cleaned)
        self.assertIn("结论：条件性成立", cleaned)

    def test_sanitize_fact_check_reply_only_relabels_conditional_point(self) -> None:
        reply = (
            "事实核查：混合结论\n"
            "1. 核查点：相关法规已经发布。\n"
            "结论：已核实。\n"
            "依据：官方文件可以确认法规存在。\n"
            "2. 核查点：私人飞机可以被视为符合绿色投资条件。\n"
            "结论：已核实。\n"
            "依据：只有满足技术筛选条件时才适用。"
        )

        cleaned = sanitize_fact_check_reply(reply)

        self.assertIn("1. 核查点：相关法规已经发布。\n结论：已核实。", cleaned)
        self.assertIn("2. 核查点：私人飞机可以被视为符合绿色投资条件。\n结论：条件性成立。", cleaned)

    def test_sanitize_fact_check_reply_splits_inline_conclusion_label(self) -> None:
        reply = (
            "1. 核查点：私人飞机可以被视为符合绿色投资条件。 结论：已核实。 "
            "依据：满足技术筛选条件才适用。"
        )

        cleaned = sanitize_fact_check_reply(reply)

        self.assertIn("1. 核查点：私人飞机可以被视为符合绿色投资条件。\n结论：条件性成立。", cleaned)
        self.assertIn("\n依据：满足技术筛选条件才适用。", cleaned)

    def test_read_image_input_bytes_rejects_untrusted_url_schemes_without_network(self) -> None:
        with self.assertRaises(ValueError):
            read_image_input_bytes(ImageInput(url="base64://QUJD"), max_bytes=10, timeout=1)

        with patch("fact_check.httpx.Client") as client:
            with self.assertRaises(ValueError):
                read_image_input_bytes(ImageInput(url="http://127.0.0.1/private.png"), max_bytes=10, timeout=1)
        client.assert_not_called()

    def test_read_image_input_bytes_rejects_hostname_resolving_to_private_ip(self) -> None:
        with (
            patch("fact_check._PUBLIC_HOST_CACHE", {}),
            patch("fact_check.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.8", 0))]),
            patch("fact_check.httpx.Client") as client,
        ):
            with self.assertRaises(ValueError):
                read_image_input_bytes(
                    ImageInput(url="https://private.example/image.png"),
                    max_bytes=10,
                    timeout=1,
                )

        client.assert_not_called()

    def test_dedupe_candidates_merges_near_duplicate_wrapped_claims(self) -> None:
        candidates = [
            ClaimCandidate("请核查：欧盟私人飞机可以被视为绿色投资是否属实？", "a", 5),
            ClaimCandidate("欧盟私人飞机可被视为符合绿色投资条件是否准确？", "b", 4),
            ClaimCandidate("请核查：另一条完全不同的事实是否属实？", "c", 3),
        ]

        deduped = dedupe_candidates(candidates, limit=3)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0].source, "a")


if __name__ == "__main__":
    unittest.main()
