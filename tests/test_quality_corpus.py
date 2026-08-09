from __future__ import annotations

import json
import unittest
from pathlib import Path

from astrbot_plugin_fact_check.fact_check import (
    IncompleteGenerationError,
    extract_text,
    sanitize_fact_check_reply,
    validate_complete_fact_check_result,
)


class QualityCorpusTests(unittest.TestCase):
    def test_anonymized_quality_cases_keep_expected_validation_behavior(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "fact_check_quality_cases.json"
        cases = json.loads(fixture.read_text(encoding="utf-8"))

        for case in cases:
            body = {
                "candidates": [{
                    "finishReason": case["finish_reason"],
                    "content": {"parts": [{"text": case["text"]}]},
                }],
            }
            with self.subTest(case=case["name"]):
                if case["valid"]:
                    validate_complete_fact_check_result(body, expected_claim_count=1)
                    cleaned = sanitize_fact_check_reply(extract_text(body))
                    for expected in case["contains"]:
                        self.assertIn(expected, cleaned)
                else:
                    with self.assertRaises(IncompleteGenerationError):
                        validate_complete_fact_check_result(body, expected_claim_count=1)


if __name__ == "__main__":
    unittest.main()
