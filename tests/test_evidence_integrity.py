from __future__ import annotations

import unittest
from typing import ClassVar

from astrbot_plugin_fact_check.evidence_mapping import (
    enforce_evidence_coverage,
    extract_claim_source_map,
)
from astrbot_plugin_fact_check.fact_check import (
    ClaimCandidate,
    IncompleteGenerationError,
    dedupe_candidates,
    salvage_partial_fact_check_reply,
    validate_complete_fact_check_result,
)


class GroundingOffsetTests(unittest.TestCase):
    claims: ClassVar[list[str]] = ["甲公司发布新手机。", "乙公司发布新电脑。"]
    source = "公司公告：https://company.example/report"

    def body(self, parts: list[str], segment: dict) -> dict:
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": part} for part in parts]},
                    "groundingMetadata": {
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://company.example/report",
                                    "title": "公司公告",
                                }
                            }
                        ],
                        "groundingSupports": [
                            {
                                "segment": segment,
                                "groundingChunkIndices": [0],
                            }
                        ],
                    },
                }
            ],
        }

    def test_utf8_offsets_do_not_move_first_claim_evidence_to_second_claim(self):
        for prefix in ("", "📱"):
            with self.subTest(prefix=prefix):
                text = (
                    prefix
                    + "事实核查：可信\n"
                    + "\n".join(
                        f"{index}. 核查点：{claim}\n结论：已核实\n"
                        f"依据：{claim}\n证据关系：支持一致"
                        for index, claim in enumerate(self.claims, 1)
                    )
                )
                start = text.index("依据：") + len("依据：")
                body = self.body(
                    [text],
                    {
                        "startIndex": len(text[:start].encode("utf-8")),
                        "endIndex": len(
                            text[: start + len(self.claims[0])].encode("utf-8")
                        ),
                        "text": self.claims[0],
                    },
                )

                mapped = extract_claim_source_map(body, self.claims)

                self.assertEqual(mapped, [[self.source], []])
                rendered = enforce_evidence_coverage(text, mapped, self.claims)
                second_block = rendered.split("2. 核查点：", 1)[1]
                self.assertIn("结论：证据不足", second_block)

    def test_offsets_are_relative_to_the_selected_part(self):
        parts = [
            f"事实核查：可信\n1. 核查点：{self.claims[0]}\n" + "背景信息。" * 30 + "\n",
            f"2. 核查点：{self.claims[1]}\n结论：已核实\n依据：公告已公开。",
        ]
        start = parts[1].index("公告已公开。")
        body = self.body(
            parts,
            {
                "partIndex": 1,
                "startIndex": len(parts[1][:start].encode("utf-8")),
                "endIndex": len(parts[1].encode("utf-8")),
                "text": "公告已公开。",
            },
        )

        self.assertEqual(
            extract_claim_source_map(body, self.claims), [[], [self.source]]
        )

    def test_invalid_byte_offsets_are_not_reassigned_using_text_overlap(self):
        text = f"事实核查：可信\n1. 核查点：{self.claims[0]}"
        for offsets in (
            {"startIndex": -1},
            {"startIndex": 1},
            {"startIndex": 9999},
            {"startIndex": 0, "endIndex": 1},
            {"startIndex": 0, "endIndex": 9999},
            {"partIndex": -1, "startIndex": 0},
            {"partIndex": 2, "startIndex": 0},
        ):
            with self.subTest(offsets=offsets):
                body = self.body([text], {"text": self.claims[0], **offsets})
                self.assertEqual(extract_claim_source_map(body, self.claims), [[], []])

    def test_missing_offsets_still_allow_text_based_grounding(self):
        body = self.body([], {"text": self.claims[1]})
        self.assertEqual(
            extract_claim_source_map(body, self.claims), [[], [self.source]]
        )


class ClaimIdentityTests(unittest.TestCase):
    changed_claims = (
        ("该药物每日服用5毫克。", "该药物每日服用50毫克。"),
        ("该政策于2025年生效。", "该政策于2026年生效。"),
        ("该公司2025年营收增长10%。", "该公司2025年营收下降10%。"),
        ("该药物每日服用0.5毫克。", "该药物每日服用5毫克。"),
        ("该药物每日服用5毫克。", "该药物每日服用5克。"),
        ("该公司2025年利润增幅为-10%。", "该公司2025年利润增幅为10%。"),
        ("该公司2025年营收增长10%。", "该公司2025年营收增长10。"),
        ("该政策于2025年生效。", "该政策已经生效。"),
    )

    def body(self, claim: str) -> dict:
        return {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    f"事实核查：可信\n1. 核查点：{claim}\n结论：已核实\n"
                                    "依据：公开公告列明了上述数值和时间。\n证据关系：支持一致"
                                )
                            }
                        ]
                    },
                }
            ]
        }

    def test_complete_result_rejects_changed_quantities_dates_and_direction(self):
        for expected, actual in self.changed_claims:
            with (
                self.subTest(expected=expected, actual=actual),
                self.assertRaises(IncompleteGenerationError),
            ):
                validate_complete_fact_check_result(
                    self.body(actual), expected_claims=[ClaimCandidate(expected)]
                )

    def test_partial_recovery_cannot_restore_an_altered_claim(self):
        for expected, actual in self.changed_claims:
            with self.subTest(expected=expected, actual=actual):
                body = self.body(actual)
                body["candidates"][0]["finishReason"] = "MAX_TOKENS"
                self.assertEqual(
                    salvage_partial_fact_check_reply(body, [ClaimCandidate(expected)]),
                    "",
                )

    def test_deduplication_keeps_distinct_decimals_and_signs(self):
        claims = [
            "该药物每日服用5.0毫克。",
            "该药物每日服用50毫克。",
            "该公司利润增幅为-10%。",
            "该公司利润增幅为10%。",
        ]
        result = dedupe_candidates([ClaimCandidate(claim) for claim in claims], limit=4)
        self.assertEqual([item.claim for item in result], claims)

    def test_equivalent_wording_and_numeric_formatting_remain_valid(self):
        for expected, actual in (
            ("请核查：该政策于2025年生效是否属实？", "该政策于2025年生效是否属实。"),
            ("该药物每日服用５ 毫克。", "该药物每日服用5毫克。"),
            ("该政策于2025年生效。", "该政策于2025年正式生效。"),
            ("该公司营收增长1,000万元。", "该公司营收增长1000万元。"),
            ("该公司营收增长10%。", "该公司营收增加10%。"),
        ):
            with self.subTest(expected=expected, actual=actual):
                validate_complete_fact_check_result(
                    self.body(actual), expected_claims=[ClaimCandidate(expected)]
                )


if __name__ == "__main__":
    unittest.main()
