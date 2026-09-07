from __future__ import annotations

import unittest
from typing import ClassVar

from astrbot_plugin_fact_check.evidence_mapping import (
    enforce_evidence_coverage,
    extract_claim_source_map,
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


if __name__ == "__main__":
    unittest.main()
