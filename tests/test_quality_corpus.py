from __future__ import annotations

import json
import unittest
from pathlib import Path

from astrbot_plugin_fact_check.evidence_mapping import enforce_evidence_coverage
from astrbot_plugin_fact_check.fact_check import (
    ClaimCandidate,
    IncompleteGenerationError,
    extract_text,
    sanitize_fact_check_reply,
    validate_complete_fact_check_result,
)


class QualityCorpusTests(unittest.TestCase):
    def test_evidence_quality_cases_enforce_conflict_and_source_strength(self) -> None:
        fixture = (
            Path(__file__).parent
            / "fixtures"
            / "fact_check_evidence_quality_cases.json"
        )
        cases = json.loads(fixture.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(cases), 20)
        for case in cases:
            reply = (
                f"事实核查：{case['summary']}\n"
                f"1. 核查点：{case['claim']}\n"
                f"结论：{case['conclusion']}\n"
                "依据：语料中的证据说明。\n"
                f"证据关系：{case['relation']}"
            )
            with self.subTest(case=case["name"]):
                rendered = enforce_evidence_coverage(
                    reply,
                    [case["sources"]],
                    [ClaimCandidate(case["claim"])],
                )
                self.assertIn(f"事实核查：{case['expect_summary']}", rendered)
                self.assertIn(f"结论：{case['expect_conclusion']}", rendered)

    def test_anonymized_quality_cases_keep_expected_validation_behavior(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "fact_check_quality_cases.json"
        cases = json.loads(fixture.read_text(encoding="utf-8"))

        for case in cases:
            body = {
                "candidates": [
                    {
                        "finishReason": case["finish_reason"],
                        "content": {"parts": [{"text": case["text"]}]},
                    }
                ],
            }
            with self.subTest(case=case["name"]):
                if case["valid"]:
                    validate_complete_fact_check_result(body, expected_claim_count=1)
                    cleaned = sanitize_fact_check_reply(extract_text(body))
                    for expected in case["contains"]:
                        self.assertIn(expected, cleaned)
                else:
                    with self.assertRaises(IncompleteGenerationError):
                        validate_complete_fact_check_result(
                            body, expected_claim_count=1
                        )


if __name__ == "__main__":
    unittest.main()
