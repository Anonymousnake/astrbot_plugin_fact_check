from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CodeHygieneTests(unittest.TestCase):
    def test_core_pipeline_uses_structured_logger_not_print(self) -> None:
        source = (ROOT / "fact_check.py").read_text(encoding="utf-8")

        self.assertNotIn("print(", source)

    def test_removed_urllib_retry_helper_does_not_return(self) -> None:
        source = (ROOT / "fact_check.py").read_text(encoding="utf-8")

        self.assertNotIn("def request_with_retry(", source)
