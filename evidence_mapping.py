from __future__ import annotations

import re
from typing import Any


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
    metadata = (response_candidates[0] if response_candidates else {}).get("groundingMetadata") or {}
    chunks = metadata.get("groundingChunks") or []
    supports = metadata.get("groundingSupports") or []
    claim_terms = [_match_terms(_claim_text(claim)) for claim in claims]
    per_claim_limit = max(1, min(3, int(limit_per_claim or 1)))

    for support in supports:
        if not isinstance(support, dict):
            continue
        segment_text = str((support.get("segment") or {}).get("text") or "").strip()
        segment_terms = _match_terms(segment_text)
        scores = [len(terms.intersection(segment_terms)) for terms in claim_terms]
        best_index = 0 if len(claims) == 1 else max(range(len(scores)), key=scores.__getitem__)
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


def append_claim_source_hints(reply: str, claim_sources: list[list[str]]) -> str:
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
        sources = claim_sources[current_claim] if current_claim < len(claim_sources) else []
        if sources:
            output.append("直接来源：" + "；".join(_source_title(source) for source in sources[:2]))
        else:
            output.append("直接来源：未找到直接支持来源")
    return "\n".join(output).strip()
