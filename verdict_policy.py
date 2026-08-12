from __future__ import annotations

import re


CLAIM_LABELS = (
    "已核实",
    "条件性成立",
    "表述需限定",
    "部分存疑",
    "不准确",
    "证据不足",
    "无法判断",
)
EVIDENCE_RELATIONS = (
    "支持一致",
    "反驳一致",
    "来源冲突",
    "仅背景",
    "无直接证据",
)
WEAK_EVIDENCE_RELATIONS = {
    "仅背景": "证据不足（现有来源仅提供背景信息）",
    "无直接证据": "证据不足（现有来源没有直接支持该命题）",
}


def claim_evidence_relations(reply: str) -> dict[int, str]:
    relations: dict[int, str] = {}
    blocks = re.split(
        r"(?=^\s*\d+\.\s*核查点[：:])",
        str(reply or ""),
        flags=re.MULTILINE,
    )
    for block in blocks:
        point = re.search(r"^\s*(\d+)\.\s*核查点[：:]", block, flags=re.MULTILINE)
        relation = re.search(
            r"^\s*证据关系[：:]\s*([^\n]+)", block, flags=re.MULTILINE
        )
        if point and relation:
            relations[int(point.group(1)) - 1] = relation.group(1).strip()
    return relations


def reconcile_fact_check_summary(reply: str) -> str:
    """Derive the overall label from the final child labels."""
    rendered = str(reply or "").strip()
    current_match = re.search(
        r"(?:^|\n)\s*事实核查[：:]\s*([^\n]+)", rendered
    )
    current = current_match.group(1).strip() if current_match else ""
    labels_pattern = "|".join(
        re.escape(label) for label in sorted(CLAIM_LABELS, key=len, reverse=True)
    )
    labels = re.findall(
        rf"^\s*结论[：:]\s*({labels_pattern})(?=$|[\s，。；;、：:(（])",
        rendered,
        flags=re.MULTILINE,
    )
    if not labels:
        return rendered

    label_set = set(labels)
    if summary_matches_claim_labels(current, labels):
        return rendered

    positive = {"已核实", "条件性成立", "表述需限定"}
    uncertain = {"证据不足", "无法判断"}
    if label_set == {"已核实"}:
        summary = "可信"
    elif label_set == {"不准确"}:
        summary = "基本不实"
    elif label_set.issubset(uncertain):
        summary = "证据不足"
    elif label_set.issubset(positive):
        summary = "基本可信但需限定"
    elif "部分存疑" in label_set:
        summary = "部分存疑"
    else:
        summary = "混合结论"

    return re.sub(
        r"(^|\n)(\s*事实核查[：:]\s*)[^\n]+",
        lambda match: match.group(1) + match.group(2) + summary,
        rendered,
        count=1,
    )


def summary_matches_claim_labels(summary: str, labels: list[str]) -> bool:
    current = str(summary or "").strip()
    label_set = set(labels)
    if not label_set:
        return False
    positive = {"已核实", "条件性成立", "表述需限定"}
    uncertain = {"证据不足", "无法判断"}
    return bool(
        (current.startswith("可信") and label_set == {"已核实"})
        or (current.startswith("基本可信但需限定") and label_set.issubset(positive))
        or (current.startswith("基本不实") and label_set == {"不准确"})
        or (current.startswith("证据不足") and label_set.issubset(uncertain))
        or current.startswith(("部分存疑", "混合结论"))
    )
