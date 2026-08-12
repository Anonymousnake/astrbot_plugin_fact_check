from __future__ import annotations

import re
from typing import Any

try:
    from .source_quality import has_strong_claim_evidence
    from .verdict_policy import (
        WEAK_EVIDENCE_RELATIONS,
        claim_evidence_relations,
        reconcile_fact_check_summary,
    )
except ImportError:  # pragma: no cover - supports direct module imports in tests
    from source_quality import has_strong_claim_evidence
    from verdict_policy import (
        WEAK_EVIDENCE_RELATIONS,
        claim_evidence_relations,
        reconcile_fact_check_summary,
    )


def _match_terms(text: str) -> set[str]:
    normalized = str(text or "").lower()
    terms = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", normalized)
        if not token.isdigit() and token not in {"the", "and", "for", "with", "from"}
    }
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        terms.add(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _claim_text(claim: Any) -> str:
    return str(getattr(claim, "claim", claim) or "").strip()


_CLAIM_MATCH_STOPWORDS = {
    "事实",
    "关于",
    "声称",
    "是否",
    "消息",
    "核查",
    "真的",
    "真假",
    "说法",
    "问题",
    "the",
    "and",
    "claim",
    "fact",
    "whether",
}


def claim_text_matches(expected: Any, actual: str) -> bool:
    """Return whether a rendered point still represents its expected claim."""
    expected_text = _claim_text(expected)
    actual_text = str(actual or "").strip()
    expected_compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", expected_text.lower())
    actual_compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", actual_text.lower())
    if not expected_compact or not actual_compact:
        return False
    if min(len(expected_compact), len(actual_compact)) >= 4 and (
        expected_compact in actual_compact or actual_compact in expected_compact
    ):
        return True
    expected_terms = _match_terms(expected_text) - _CLAIM_MATCH_STOPWORDS
    actual_terms = _match_terms(actual_text) - _CLAIM_MATCH_STOPWORDS
    if not expected_terms or not actual_terms:
        return False
    shared = expected_terms.intersection(actual_terms)
    coverage = len(shared) / max(1, min(len(expected_terms), len(actual_terms)))
    return len(shared) >= 2 and coverage >= 0.45


_EVIDENCE_REQUIRED_LABELS = (
    "已核实",
    "条件性成立",
    "表述需限定",
    "部分存疑",
    "不准确",
)

_HIGH_RISK_CLAIM_RE = re.compile(
    r"(法规|政策|法律|违法|法院|医学|疾病|治疗|药物|疫苗|金融|投资|证券|安全|事故|伤亡|"
    r"policy|law|legal|court|medical|disease|treatment|drug|vaccine|finance|investment|security|accident|injury)",
    flags=re.IGNORECASE,
)


def evidence_text_relevant(claim: Any, evidence_text: str) -> bool:
    """Reject extracted pages that do not materially overlap the checked claim."""
    claim_text = re.sub(r"^\s*请核查[：:]?\s*", "", _claim_text(claim))
    claim_text = re.sub(r"是否属实[？?]?\s*$", "", claim_text)
    claim_terms = _match_terms(claim_text) - _CLAIM_MATCH_STOPWORDS
    evidence_terms = _match_terms(evidence_text) - _CLAIM_MATCH_STOPWORDS
    if not claim_terms or not evidence_terms:
        return False
    shared = claim_terms.intersection(evidence_terms)
    minimum = 1 if len(claim_terms) <= 2 else 2
    return len(shared) >= minimum and len(shared) / max(1, len(claim_terms)) >= 0.2


def enforce_evidence_coverage(
    reply: str,
    claim_sources: list[list[str]],
    claims: list[Any] | None = None,
) -> str:
    """Downgrade decisive claim labels that have no directly mapped source."""
    if not reply:
        return reply
    output: list[str] = []
    current_claim: int | None = None
    claim_count = 0
    downgraded = 0
    relations = claim_evidence_relations(reply)
    label_pattern = "|".join(
        re.escape(label)
        for label in sorted(_EVIDENCE_REQUIRED_LABELS, key=len, reverse=True)
    )
    for line in str(reply).splitlines():
        point_match = re.match(r"^\s*(\d+)\.\s*核查点[：:]", line)
        if point_match:
            current_claim = int(point_match.group(1)) - 1
            claim_count = max(claim_count, current_claim + 1)
        conclusion = re.match(
            rf"^(\s*结论[：:]\s*)(?:{label_pattern})(?:[^\n]*)$", line
        )
        has_sources = bool(
            current_claim is not None
            and current_claim < len(claim_sources)
            and claim_sources[current_claim]
        )
        if conclusion:
            claim = (
                claims[current_claim]
                if claims and current_claim is not None and current_claim < len(claims)
                else ""
            )
            sources = (
                claim_sources[current_claim]
                if current_claim is not None and current_claim < len(claim_sources)
                else []
            )
            relation = relations.get(current_claim, "")
            if relation.startswith("来源冲突"):
                line = conclusion.group(1) + "部分存疑（直接来源之间存在冲突）"
                downgraded += 1
            elif weak_reason := next(
                (
                    reason
                    for label, reason in WEAK_EVIDENCE_RELATIONS.items()
                    if relation.startswith(label)
                ),
                "",
            ):
                line = conclusion.group(1) + weak_reason
                downgraded += 1
            elif not has_sources:
                line = conclusion.group(1) + "证据不足（未找到可核验的直接来源）"
                downgraded += 1
            elif _HIGH_RISK_CLAIM_RE.search(
                _claim_text(claim)
            ) and not has_strong_claim_evidence(sources):
                line = (
                    conclusion.group(1)
                    + "证据不足（高风险命题缺少官方来源或多源交叉验证）"
                )
                downgraded += 1
        output.append(line)
    rendered = "\n".join(output).strip()
    return reconcile_fact_check_summary(rendered)


def merge_claim_sources(
    *source_maps: list[list[str]],
    limit_per_claim: int = 2,
) -> list[list[str]]:
    """Merge per-claim provenance without losing claim alignment."""
    claim_count = max((len(source_map) for source_map in source_maps), default=0)
    merged: list[list[str]] = [[] for _ in range(claim_count)]
    per_claim_limit = max(1, min(5, int(limit_per_claim or 1)))
    for source_map in source_maps:
        for index, sources in enumerate(source_map):
            for source in sources:
                clean_source = str(source or "").strip()
                if clean_source and clean_source not in merged[index]:
                    merged[index].append(clean_source)
                if len(merged[index]) >= per_claim_limit:
                    break
    return merged


def extract_claim_source_map(
    body: dict[str, Any],
    claims: list[Any],
    *,
    limit_per_claim: int = 2,
) -> list[list[str]]:
    """Map Gemini grounding support segments to the claims they overlap."""
    mapped: list[list[str]] = [[] for _ in claims]
    if not claims:
        return mapped
    response_candidates = body.get("candidates", []) or []
    metadata = (response_candidates[0] if response_candidates else {}).get(
        "groundingMetadata"
    ) or {}
    chunks = metadata.get("groundingChunks") or []
    supports = metadata.get("groundingSupports") or []
    claim_terms = [_match_terms(_claim_text(claim)) for claim in claims]
    per_claim_limit = max(1, min(3, int(limit_per_claim or 1)))
    content = (response_candidates[0] if response_candidates else {}).get(
        "content"
    ) or {}
    response_text = "".join(
        str(part.get("text") or "")
        for part in content.get("parts") or []
        if isinstance(part, dict)
    )
    point_matches = list(
        re.finditer(r"^\s*(\d+)\.\s*核查点[：:]", response_text, flags=re.MULTILINE),
    )

    for support in supports:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment") or {}
        segment_text = str(segment.get("text") or "").strip()
        segment_start = segment.get("startIndex")
        best_index: int | None = None
        if isinstance(segment_start, int) and point_matches:
            for point_index, point_match in enumerate(point_matches):
                next_start = (
                    point_matches[point_index + 1].start()
                    if point_index + 1 < len(point_matches)
                    else len(response_text)
                )
                if point_match.start() <= segment_start < next_start:
                    candidate_index = int(point_match.group(1)) - 1
                    if 0 <= candidate_index < len(claims):
                        best_index = candidate_index
                    break
        if best_index is None:
            segment_terms = _match_terms(segment_text)
            scores = [len(terms.intersection(segment_terms)) for terms in claim_terms]
            best_index = (
                0
                if len(claims) == 1
                else max(range(len(scores)), key=scores.__getitem__)
            )
            if len(claims) > 1 and not scores[best_index]:
                continue
        for index in support.get("groundingChunkIndices") or []:
            if not isinstance(index, int) or not 0 <= index < len(chunks):
                continue
            web = (chunks[index] or {}).get("web") or {}
            uri = str(web.get("uri") or "").strip()
            title = str(web.get("title") or "").strip()
            if not uri:
                continue
            source = f"{title or uri}：{uri}"
            if source not in mapped[best_index]:
                mapped[best_index].append(source)
            if len(mapped[best_index]) >= per_claim_limit:
                break
    return mapped


def has_grounding_supports(body: dict[str, Any]) -> bool:
    candidates = body.get("candidates", []) or []
    metadata = (candidates[0] if candidates else {}).get("groundingMetadata") or {}
    return bool(metadata.get("groundingSupports"))


def _source_title(source: str) -> str:
    title, separator, remainder = str(source or "").partition("：")
    if separator and remainder.startswith(("http://", "https://")) and title.strip():
        return title.strip()
    return str(source or "").strip()


def _source_url(source: str) -> str:
    match = re.search(r"https?://[^\s；;，,]+", str(source or ""))
    return match.group(0).rstrip(".。)]）") if match else ""


def _citation_index(source: str, references: list[str]) -> int | None:
    source_url = _source_url(source)
    source_title = _source_title(source)
    for index, reference in enumerate(references, start=1):
        reference_url = _source_url(reference)
        if source_url and reference_url and source_url == reference_url:
            return index
        if not source_url and source_title and source_title == _source_title(reference):
            return index
    return None


def append_claim_source_hints(
    reply: str,
    claim_sources: list[list[str]],
    references: list[str] | None = None,
) -> str:
    """Add direct-support provenance under each structured claim block."""
    if not reply or not claim_sources:
        return reply
    output: list[str] = []
    current_claim: int | None = None
    for line in str(reply).splitlines():
        point_match = re.match(r"^\s*(\d+)\.\s*核查点[：:]", line)
        if point_match:
            current_claim = int(point_match.group(1)) - 1
        output.append(line)
        if current_claim is None or not re.match(r"^\s*依据[：:]", line):
            continue
        sources = (
            claim_sources[current_claim] if current_claim < len(claim_sources) else []
        )
        if sources:
            labels: list[str] = []
            for source in sources[:2]:
                citation = _citation_index(source, references or [])
                prefix = f"[{citation}] " if citation is not None else ""
                labels.append(prefix + _source_title(source))
            output.append("直接来源：" + "；".join(labels))
        else:
            output.append("直接来源：未找到直接支持来源")
    return "\n".join(output).strip()
