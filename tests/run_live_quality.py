"""Evaluate public, source-checked claims through the live fact-check pipeline."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from ..fact_check import FactCheckRequest, run_fact_check
from ..pipeline_config import build_fact_check_kwargs
from ..verdict_policy import CLAIM_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    fixture = Path(__file__).parent / "fixtures" / "fact_check_live_cases.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    unknown = set(args.case) - {case["id"] for case in cases}
    if unknown:
        parser.error(f"Unknown case ids: {', '.join(sorted(unknown))}")
    selected = [case for case in cases if not args.case or case["id"] in args.case]
    results = []
    for case in selected:
        started = time.monotonic()
        print(f"Evaluating {case['id']}", flush=True)
        try:
            result = run_fact_check(
                **build_fact_check_kwargs(
                    config,
                    FactCheckRequest(text=case["claim"], trigger_text="/事实核查"),
                    args.timeout,
                    list_config=lambda key, default: config.get(key) or default,
                )
            )
            labels = [
                next(
                    (label for label in CLAIM_LABELS if value.startswith(label)),
                    "unknown",
                )
                for value in re.findall(
                    r"^\s*结论[：:]\s*([^\n]+)", result.reply, re.MULTILINE
                )
            ]
            row = {
                "id": case["id"],
                "expected": case["expected_conclusion"],
                "labels": labels,
                "verdict_matches": bool(labels)
                and all(label == case["expected_conclusion"] for label in labels),
                "complete": result.reason.startswith("ok")
                and not result.reason.startswith("ok; partial"),
                "sources": result.sources,
                "reply": result.reply,
            }
        except Exception as exc:  # noqa: BLE001 - keep evaluating after one provider failure
            row = {
                "id": case["id"],
                "verdict_matches": False,
                "complete": False,
                "sources": [],
                "error": type(exc).__name__,
            }
        row["elapsed_seconds"] = round(time.monotonic() - started, 2)
        results.append(row)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"reply", "sources"}
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    report = {
        "evaluated": len(results),
        "verdict_matches": sum(row["verdict_matches"] for row in results),
        "complete": sum(row["complete"] for row in results),
        "with_sources": sum(bool(row["sources"]) for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))
    raise SystemExit(
        0
        if all(
            row["verdict_matches"] and row["complete"] and row["sources"]
            for row in results
        )
        else 1
    )


if __name__ == "__main__":
    main()
