from __future__ import annotations

import base64
import io
import ipaddress
import json
import os
import random
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from PIL import Image


TRIGGER_RE = re.compile(
    r"(?:^|[\s\u00a0\u200b\u200c\u200d/])(?:事实核查|factcheck|fact-check)(?:[\s\u00a0\u200b\u200c\u200d]*|$|[:：])",
    re.IGNORECASE,
)
NO_CHECKABLE_CLAIM = "无明确事实断言"
FAILED_REPLY = "这条我现在没查成。"
LIGHTWEIGHT_MODELS = {"gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview"}
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
ANYSEARCH_DEFAULT_ENDPOINT = "https://api.anysearch.com/mcp"
ANYSEARCH_CONTENT_TYPES = {"web", "news", "doc", "academic", "data"}
ANYSEARCH_FRESHNESS_VALUES = {"day", "week", "month", "year"}
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
URL_RE = re.compile(r"(?:-\s*\*\*URL\*\*:\s*)?(https?://[^\s<>\]\)\"']+)", re.IGNORECASE)
META_CLAIM_RE = re.compile(
    r"(系统自动生成|无需核查|不需要核查|不用核查|无法核查|没有必要核查|"
    r"此问题|该问题|这个问题|本问题|用户请求|机器人|bot|工具调用|"
    r"事实核查命令|核查指令|不是事实断言|无事实断言)",
    re.IGNORECASE,
)
HIGH_RISK_COMPOSITE_EXTRACTION_RULES = """\
- 对法规、政策、医学、法律、金融、安全等高风险复合命题要拆成 atomic claims，不要只保留一个笼统问题。
- 如果原文同时声称“某法规/政策存在”和“某具体产品、硬件、功能、销售行为、违法性会被该法规直接覆盖”，至少拆成两个问题：
  1. 请核查相关法规/政策是否存在、发布机构和生效时间是否属实？
  2. 请核查该法规/政策是否明确覆盖原文提到的具体产品、硬件、功能、销售行为或违法性结论？
- 不要过度细拆普通事实；只有这类“基础事实 + 具体适用/推论”的高风险命题才强制拆分。"""
HIGH_RISK_VERDICT_CALIBRATION_RULES = """\
- 证据必须直接覆盖核心争议命题才可判“大致可信”；只证实外围事实或基础事实不够。
- 对“法规/政策存在 + 具体产品、硬件、功能、销售、违法性推论”的复合命题，必须分别判断：
  1. 法规/政策是否真实存在、发布机构和生效时间是否属实；
  2. 条文、官方解释或权威报道是否直接说明原文中的具体对象/行为被覆盖。
- 若来源只支持法规/政策存在，但没有直接支持具体适用推论，整体结论最高为“部分存疑”，并明确写“未找到直接证据支持该具体推论”。
- 总结论按最关键争议命题决定，不要因为任一子事实成立就把整体判成“大致可信”。
- 要点中尽量标出“已证实 / 未直接证实 / 存疑”。"""
CONDITIONAL_CLAIM_RE = re.compile(
    r"(可以被视为|可被视为|符合.+条件|满足.+条件|在满足.+条件下|"
    r"eligible|aligned|taxonomy[- ]?(?:compatible|eligible|aligned))",
    re.IGNORECASE,
)


def current_time_context() -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return (
        f"当前日期时间：{now:%Y-%m-%d %H:%M:%S %Z}；"
        f"今天是{now:%Y年%m月%d日}，星期{now.isoweekday()}；"
        f"Unix时间戳：{int(now.timestamp())}。"
    )


@dataclass(slots=True)
class ImageInput:
    url: str
    file_name: str = ""
    path: str = ""


@dataclass(slots=True)
class FactCheckRequest:
    text: str
    trigger_text: str
    speaker: str = ""
    images: list[ImageInput] = field(default_factory=list)


@dataclass(slots=True)
class ClaimCandidate:
    claim: str
    source: str = ""
    priority: int = 3


@dataclass(slots=True)
class FactCheckResult:
    reply: str
    reason: str = ""
    sources: list[str] = field(default_factory=list)
    candidates: list[ClaimCandidate] = field(default_factory=list)


@dataclass(slots=True)
class AnysearchEvidence:
    text: str = ""
    sources: list[str] = field(default_factory=list)
    reason: str = ""


def is_trigger(text: str) -> bool:
    return bool(TRIGGER_RE.search(_normalize(text)))


def remove_trigger(text: str) -> str:
    return TRIGGER_RE.sub(" ", _normalize(text)).strip(" \t\r\n:：?")


def explain_failure(reason: str) -> str:
    lowered = str(reason or "").lower()
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        detail = "API 在限流。"
    elif "api key" in lowered or "unauthorized" in lowered or "401" in lowered:
        detail = "Gemini key 没配好或被拒了。"
    elif "no quoted text" in lowered or "no inline claim" in lowered:
        detail = "我没拿到要核查的原文。"
    elif "image" in lowered and ("download" in lowered or "too large" in lowered):
        detail = "图片没下载成或太大了。"
    elif "no checkable" in lowered:
        detail = "我没从里面抽出能查的事实断言。"
    elif "empty" in lowered:
        detail = "模型返回了空结果。"
    elif "timeout" in lowered:
        detail = "请求超时了。"
    elif "ssl" in lowered or "eof" in lowered or "connection" in lowered:
        detail = "网络连接中途断了。"
    else:
        detail = "请求过程里出了点问题。"
    return f"{FAILED_REPLY}\n原因：{detail}"


def run_fact_check(
    *,
    request_data: FactCheckRequest,
    api_key: str,
    base_url: str,
    pre_model: str,
    main_models: list[str],
    max_image_bytes: int = 5 * 1024 * 1024,
    long_image_chunk_height: int = 2200,
    long_image_max_parts: int = 8,
    long_image_max_width: int = 1280,
    image_download_timeout: int = 10,
    pre_request_timeout: int = 25,
    main_request_timeout: int = 45,
    anysearch_enabled: bool = False,
    anysearch_endpoint: str = ANYSEARCH_DEFAULT_ENDPOINT,
    anysearch_api_key: str = "",
    anysearch_timeout: int = 20,
    anysearch_max_claims: int = 3,
    anysearch_max_results_per_claim: int = 3,
    anysearch_extract_top_urls: int = 2,
    anysearch_max_chars: int = 6000,
    anysearch_freshness: str = "",
    anysearch_content_types: list[str] | None = None,
) -> FactCheckResult:
    api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return FactCheckResult(FAILED_REPLY, "missing Gemini API key")

    candidates: list[ClaimCandidate] = []
    text_context = request_data.text.strip()
    if text_context:
        print(f"[astrbot-fact-check-stage] text-preprocess start len={len(text_context)} model={pre_model}", flush=True)
        candidates.extend(
            extract_claims_from_text(
                text_context,
                model=pre_model,
                api_key=api_key,
                base_url=base_url,
                request_timeout=pre_request_timeout,
            ),
        )
        print(f"[astrbot-fact-check-stage] text-preprocess done candidates={len(candidates)}", flush=True)

    if request_data.images:
        print(
            "[astrbot-fact-check-stage] image-preprocess start "
            f"images={len(request_data.images)} model={pre_model}",
            flush=True,
        )
        candidates.extend(
            extract_claims_from_images(
                request_data.images,
                context_text=text_context,
                model=pre_model,
                api_key=api_key,
                base_url=base_url,
                max_image_bytes=max_image_bytes,
                long_image_chunk_height=long_image_chunk_height,
                long_image_max_parts=long_image_max_parts,
                long_image_max_width=long_image_max_width,
                image_download_timeout=image_download_timeout,
                request_timeout=pre_request_timeout,
            ),
        )
        print(f"[astrbot-fact-check-stage] image-preprocess done candidates={len(candidates)}", flush=True)

    if not candidates:
        if text_context:
            candidates.append(
                ClaimCandidate(
                    claim=f"请核查下面聊天内容中涉及的事实是否准确：{text_context[:800]}",
                    source="原始聊天内容",
                    priority=2,
                ),
            )
        else:
            return FactCheckResult(
                FAILED_REPLY,
                f"no checkable factual claims; text={text_context[:120]}; images={len(request_data.images)}",
                [],
                [],
            )

    deduped = dedupe_candidates(candidates, limit=3)
    candidate_text = format_candidates(deduped)
    anysearch_evidence = collect_anysearch_evidence(
        deduped,
        enabled=anysearch_enabled,
        endpoint=anysearch_endpoint,
        api_key=anysearch_api_key,
        timeout=anysearch_timeout,
        max_claims=anysearch_max_claims,
        max_results_per_claim=anysearch_max_results_per_claim,
        extract_top_urls=anysearch_extract_top_urls,
        max_chars=anysearch_max_chars,
        freshness=anysearch_freshness,
        content_types=anysearch_content_types,
    )
    if anysearch_evidence.reason:
        print(f"[astrbot-fact-check-anysearch] {anysearch_evidence.reason}", flush=True)
    evidence_block = (
        "\nAnysearch 预检索证据（仅作线索，最终以可核验来源为准）：\n"
        f"{sanitize_anysearch_evidence_text(anysearch_evidence.text)}\n"
        if anysearch_evidence.text
        else ""
    )
    speaker_line = f"发言人：{request_data.speaker}\n" if request_data.speaker else ""
    main_image_parts = build_inline_image_parts(
        request_data.images,
        max_image_bytes=max_image_bytes,
        long_image_chunk_height=long_image_chunk_height,
        long_image_max_parts=long_image_max_parts,
        long_image_max_width=long_image_max_width,
        image_download_timeout=image_download_timeout,
        stage="main-reference",
    )
    image_line = (
        f"参考图片：已附上 {len(main_image_parts)} 张原图。请同时参考图中文字、截图上下文和画面含义。\n"
        if main_image_parts
        else "参考图片：原图未成功附上，只能依据前置整理出的核查问题和原始文字判断。\n"
    )
    prompt = f"""你是一个中文事实核查助手。请使用 Google Search grounding 核查下面的聊天内容和核查问题列表。

时间上下文：
{current_time_context()}

关键规则：
- 所有“今天、昨天、明天、尚未发生、已经发布、即将发布”等时间判断，必须以上面的当前日期时间为准。
- 如果图片、网页或搜索结果中出现发布日期/发布时间，必须先与当前日期比较；不要使用模型训练截止日期或内置知识作为当前时间。
- 若声称某日期“尚未到来”，必须确认该日期确实晚于当前日期；否则不要这样判断。
- 如果提供了 Anysearch 预检索证据，它只是辅助线索；需要和 Google Search grounding、原始图片/文字一起交叉核对。
- 若预检索摘要与更权威、更新时间更明确的来源冲突，优先依据权威来源，并说明不确定点。
{HIGH_RISK_VERDICT_CALIBRATION_RULES}

{speaker_line}{image_line}原始聊天内容：
{request_data.text or "（无文字，主要来自图片）"}

待核查问题：
{candidate_text}
{evidence_block}

输出要求：
- 中文，适合 QQ 群聊，简短但有用。
- 不要使用 Markdown 粗体、Markdown 标题、代码块或裸列表符号。
- 先给总结论，只能从这些标签中选择：可信 / 基本可信但需限定 / 条件性成立 / 混合结论 / 部分存疑 / 证据不足 / 基本不实 / 表述不准确。
- 每个核查点都必须单独写“结论：...”，子问题结论只能从这些标签中选择：已核实 / 条件性成立 / 表述需限定 / 部分存疑 / 证据不足 / 不准确 / 无法判断。
- 如果原文是“可以被视为符合某条件”“在满足条件下适用”“eligible”“aligned”“taxonomy-compatible”这类条件性表述，不要写“已证实”；优先写“条件性成立”或“表述需限定”，并说明条件。
- 证据不足就明确说不确定，不要硬判。
- 对复合命题逐项写清“已证实 / 未直接证实 / 存疑”，不要只回答其中一个子事实。
- 不要编造来源。

格式：
事实核查：<总结论>
1. 核查点：...
结论：...
依据：...
2. 核查点：...
结论：...
依据：...
来源：列出你实际用到的来源标题或站点，最多 3 个。
"""
    print(
        "[astrbot-fact-check-stage] main-check start "
        f"models={','.join(main_models)} candidates={len(deduped)}",
        flush=True,
    )
    body, used_model = generate_with_fallback(
        prompt=prompt,
        models=main_models,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_output_tokens=768,
        grounding=True,
        extra_parts=main_image_parts,
        request_timeout=main_request_timeout,
    )
    print(f"[astrbot-fact-check-stage] main-check done model={used_model}", flush=True)
    reply = sanitize_fact_check_reply(extract_text(body).strip())
    sources = dedupe_sources(extract_sources(body) + anysearch_evidence.sources, limit=5)
    if sources and "来源" not in reply:
        reply += "\n来源：" + "；".join(sources[:3])
    reply = append_anysearch_source_note(reply, anysearch_evidence)
    if used_model in LIGHTWEIGHT_MODELS and reply:
        reply += "\n（主模型繁忙，已用轻量模型核查）"
    if not reply:
        claims = "；".join(item.claim[:80] for item in deduped[:2])
        return FactCheckResult(
            f"{FAILED_REPLY}\n待查点：{claims}" if claims else FAILED_REPLY,
            f"main model returned empty reply; model={used_model}; candidates={len(deduped)}",
            sources,
            deduped,
        )
    return FactCheckResult(
        reply=reply,
        reason="ok; extracted_claims=" + "; ".join(item.claim[:80] for item in deduped),
        sources=sources,
        candidates=deduped,
    )


def append_anysearch_source_note(reply: str, evidence: AnysearchEvidence) -> str:
    sources = dedupe_sources(evidence.sources, limit=3)
    if not sources:
        return reply
    compact_sources = "；".join(compact_source_label(source) for source in sources)
    note = f"预检索线索：Anysearch 命中 {len(sources)} 个公开来源：{compact_sources}。"
    text = str(reply or "").rstrip()
    if not text:
        return note
    if "预检索线索：" in text:
        return text
    return f"{text}\n{note}"


def run_fact_check_followup(
    *,
    original_text: str,
    candidates: list[ClaimCandidate],
    previous_reply: str,
    previous_sources: list[str],
    question: str,
    api_key: str,
    base_url: str,
    main_models: list[str],
    request_timeout: int = 45,
) -> FactCheckResult:
    api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return FactCheckResult(FAILED_REPLY, "missing Gemini API key")

    candidate_text = format_candidates(candidates) if candidates else "（上次未保存待核查问题）"
    source_text = "\n".join(f"- {source}" for source in previous_sources[:5]) or "（上次未提取到来源）"
    prompt = f"""你是中文事实核查追问助手。用户正在追问上一轮事实核查结果。

时间上下文：
{current_time_context()}

关键规则：
- 所有时间判断必须以上面的当前日期时间为准。
- 不要使用模型训练截止日期或内置知识作为当前时间。

原始聊天内容：
{original_text or "（无文字或主要来自图片）"}

上次待核查问题：
{candidate_text}

上次核查结果：
{previous_reply or "（无）"}

上次来源：
{source_text}

用户追问：
{question}

要求：
- 只回答这次追问，不要完整重复上一轮事实核查。
- 必要时继续使用 Google Search grounding 查证。
- 如果新增证据会改变上次结论，请明确说“原结论需要修正”；否则说“原结论暂不改变”。
- 证据不足就说不确定，不要硬判。
- 不要编造来源。

格式：
追问结论：...
补充依据：...
是否改变原结论：...
来源：列出实际用到的来源标题或站点，最多 3 个。
"""
    body, used_model = generate_with_fallback(
        prompt=prompt,
        models=main_models,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_output_tokens=640,
        grounding=True,
        request_timeout=request_timeout,
    )
    reply = sanitize_fact_check_reply(extract_text(body).strip())
    sources = extract_sources(body)
    if sources and "来源" not in reply:
        reply += "\n来源：" + "；".join(sources)
    if used_model in LIGHTWEIGHT_MODELS and reply:
        reply += "\n（主模型繁忙，已用轻量模型回答追问）"
    if not reply:
        return FactCheckResult(
            FAILED_REPLY,
            f"follow-up model returned empty reply; model={used_model}",
            sources,
            candidates,
        )
    return FactCheckResult(reply=reply, reason="ok; follow-up", sources=sources, candidates=candidates)


def extract_claims_from_text(
    text: str,
    *,
    model: str,
    api_key: str,
    base_url: str,
    limit: int = 5,
    request_timeout: int = 25,
) -> list[ClaimCandidate]:
    prompt = f"""你是事实核查的前置整理模块。

时间上下文：
{current_time_context()}

任务：不要判断真伪。请把聊天内容整理成 0-{limit} 个适合交给联网核查模型的问题。

规则：
- 明确断言改写成“请核查：...是否属实？”
- 半句话、转述、截图上下文，保留原话和必要背景。
- 优先保留人名、地点、机构、时间、数据、事件、标题、引用。
- 只有完全没有可查信息才输出 []。
- 不要输出关于“是否需要核查”“系统自动生成”“工具/机器人提示”“用户请求事实核查”的元问题。
{HIGH_RISK_COMPOSITE_EXTRACTION_RULES}

只输出 JSON 数组：
[
  {{"question":"给大模型的核查问题","source":"用户文字/引用消息","priority":1-5}}
]

聊天内容：
{text}
"""
    body, _ = generate_with_fallback(
        prompt=prompt,
        models=[model],
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_output_tokens=640,
        grounding=False,
        request_timeout=request_timeout,
    )
    return parse_candidates(extract_text(body), limit=limit)


def extract_claims_from_images(
    images: list[ImageInput],
    *,
    context_text: str,
    model: str,
    api_key: str,
    base_url: str,
    max_image_bytes: int,
    long_image_chunk_height: int,
    long_image_max_parts: int,
    long_image_max_width: int,
    limit: int = 5,
    image_download_timeout: int = 10,
    request_timeout: int = 25,
) -> list[ClaimCandidate]:
    parts: list[dict[str, Any]] = [
        {
            "text": (
                "你是事实核查的图片前置整理模块。\n"
                f"时间上下文：{current_time_context()}\n"
                "整理图片中的日期、发布时间、相对时间时，必须保留原文日期，并让后续联网核查模型按当前日期比较。\n"
                "任务：先理解图片，再把图片内容整理成适合交给联网核查模型的问题。不要判断真伪。\n"
                "请尽量 OCR 图中文字，包括标题、表格、聊天截图、社交媒体截图、水印、来源名。\n"
                "如果图片是新闻、截图、谣言、通知、数据图、对话记录，通常至少整理出 1 个问题。\n"
                "只有纯表情包、纯风景、完全看不清、或没有任何可查信息时才输出 []。\n"
                "不要输出关于“是否需要核查”“系统自动生成”“工具/机器人提示”“用户请求事实核查”的元问题。\n"
                f"{HIGH_RISK_COMPOSITE_EXTRACTION_RULES}\n"
                "只输出 JSON 数组，格式："
                '[{"question":"给大模型的核查问题","source":"图片OCR/图片含义/用户文字+图片","priority":1-5}]\n'
                f"用户附带文字：{context_text or '无'}"
            ),
        },
    ]
    seen_image_payloads: set[str] = set()
    for item in images:
        try:
            append_unique_inline_parts(
                parts,
                download_image_as_inline_parts(
                    item,
                    max_bytes=max_image_bytes,
                    long_image_chunk_height=long_image_chunk_height,
                    long_image_max_parts=long_image_max_parts,
                    long_image_max_width=long_image_max_width,
                    timeout=image_download_timeout,
                ),
                seen_image_payloads,
                stage="preprocess",
                label=item.file_name or item.url or item.path,
            )
        except Exception as exc:
            print(f"[astrbot-fact-check-image-download-error] {item.file_name or item.url}: {exc!r}", flush=True)
    if len(parts) == 1:
        return []
    raw = call_gemini_parts(
        parts=parts,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_output_tokens=640,
        request_timeout=request_timeout,
    )
    return parse_candidates(raw, limit=limit)


def generate_with_fallback(
    *,
    prompt: str,
    models: list[str],
    api_key: str,
    base_url: str,
    temperature: float,
    max_output_tokens: int,
    grounding: bool,
    extra_parts: list[dict[str, Any]] | None = None,
    max_attempts: int | None = None,
    request_timeout: int = 45,
) -> tuple[dict[str, Any], str]:
    clean_models = [model.strip() for model in models if str(model or "").strip()]
    if not clean_models:
        raise RuntimeError("no fact-check model configured")
    attempts = max_attempts or max(1, min(3, len(clean_models)))
    last_error: Exception | None = None
    last_model = clean_models[0]
    for attempt in range(attempts):
        model = clean_models[min(attempt, len(clean_models) - 1)]
        last_model = model
        try:
            return gemini_generate(
                prompt=prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                grounding=grounding,
                extra_parts=extra_parts,
                request_timeout=request_timeout,
            ), model
        except Exception as exc:
            last_error = exc
            if not is_retryable(exc) or attempt >= attempts - 1:
                break
            next_model = clean_models[min(attempt + 1, len(clean_models) - 1)]
            wait = backoff_seconds(attempt, exc)
            print(
                "[astrbot-fact-check-retry] "
                f"attempt={attempt + 1}/{attempts} model={model} next={next_model} "
                f"wait={wait:.1f}s error={error_label(exc)}"
                ,
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"fact-check request failed after {attempts} attempt(s); "
        f"last_model={last_model}; last_error={error_label(last_error)}"
    )


def gemini_generate(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_output_tokens: int,
    grounding: bool,
    extra_parts: list[dict[str, Any]] | None = None,
    request_timeout: int = 45,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if model.startswith("gemini-2.5-flash"):
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}
    parts: list[dict[str, Any]] = [{"text": prompt}]
    if extra_parts:
        parts.extend(extra_parts)
    payload: dict[str, Any] = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }
    if grounding:
        payload["tools"] = [{"google_search": {}}]
    return post_json_with_timeout(
        base_url.rstrip("/") + f"/{model}:generateContent",
        payload,
        api_key=api_key,
        timeout=request_timeout,
    )


def call_gemini_parts(
    *,
    parts: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout: int = 25,
) -> str:
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    body = post_json_with_timeout(
        base_url.rstrip("/") + f"/{model}:generateContent",
        payload,
        api_key=api_key,
        timeout=request_timeout,
    )
    return extract_text(body)


def build_inline_image_parts(
    images: list[ImageInput],
    *,
    max_image_bytes: int,
    long_image_chunk_height: int,
    long_image_max_parts: int,
    long_image_max_width: int,
    image_download_timeout: int,
    stage: str,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    seen_image_payloads: set[str] = set()
    for item in images:
        try:
            append_unique_inline_parts(
                parts,
                download_image_as_inline_parts(
                    item,
                    max_bytes=max_image_bytes,
                    long_image_chunk_height=long_image_chunk_height,
                    long_image_max_parts=long_image_max_parts,
                    long_image_max_width=long_image_max_width,
                    timeout=image_download_timeout,
                ),
                seen_image_payloads,
                stage=stage,
                label=item.file_name or item.url or item.path,
            )
        except Exception as exc:
            print(
                f"[astrbot-fact-check-image-{stage}-error] "
                f"{item.file_name or item.url}: {exc!r}",
                flush=True,
            )
    if parts:
        print(
            f"[astrbot-fact-check-image-{stage}] attached={len(parts)}",
            flush=True,
        )
    return parts


def append_unique_inline_parts(
    target: list[dict[str, Any]],
    new_parts: list[dict[str, Any]],
    seen_payloads: set[str],
    *,
    stage: str,
    label: str,
) -> None:
    for part in new_parts:
        payload = ((part.get("inline_data") or {}).get("data") or "").strip()
        if payload and payload in seen_payloads:
            print(f"[astrbot-fact-check-image-{stage}-dedupe] skipped duplicate {label}", flush=True)
            continue
        if payload:
            seen_payloads.add(payload)
        target.append(part)


def download_image_as_inline_parts(
    item: ImageInput,
    *,
    max_bytes: int,
    long_image_chunk_height: int,
    long_image_max_parts: int,
    long_image_max_width: int,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    ref = item.path or item.url
    print(f"[astrbot-fact-check-image-download] start {item.file_name or ref}", flush=True)
    body, content_type = read_image_input_bytes(item, max_bytes=None, timeout=timeout)
    print(
        f"[astrbot-fact-check-image-download] done bytes={len(body)} source={'local' if item.path else 'remote'}",
        flush=True,
    )
    if len(body) <= max_bytes:
        return [
            make_inline_image_part(
                body,
                mime_type=guess_mime_type(item.file_name, content_type),
            ),
        ]
    return split_large_image_as_inline_parts(
        body,
        source_label=item.file_name or ref,
        max_bytes=max_bytes,
        chunk_height=long_image_chunk_height,
        max_parts=long_image_max_parts,
        max_width=long_image_max_width,
    )


def make_inline_image_part(body: bytes, *, mime_type: str) -> dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(body).decode("ascii"),
        },
    }


def split_large_image_as_inline_parts(
    body: bytes,
    *,
    source_label: str,
    max_bytes: int,
    chunk_height: int,
    max_parts: int,
    max_width: int,
) -> list[dict[str, Any]]:
    chunk_height = max(800, int(chunk_height or 2200))
    max_parts = max(1, int(max_parts or 8))
    max_width = max(320, int(max_width or 1280))
    with Image.open(io.BytesIO(body)) as image:
        image.load()
        width, height = image.size
        if width > max_width:
            new_height = max(1, round(height * (max_width / width)))
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            width, height = image.size
        total_parts = (height + chunk_height - 1) // chunk_height
        if total_parts > max_parts:
            chunk_height = (height + max_parts - 1) // max_parts
            total_parts = max_parts
        chunks: list[dict[str, Any]] = []
        for index in range(total_parts):
            top = index * chunk_height
            bottom = min(height, top + chunk_height)
            chunk = image.crop((0, top, width, bottom))
            encoded = encode_image_chunk_under_limit(chunk, max_bytes=max_bytes)
            chunks.append(make_inline_image_part(encoded, mime_type="image/jpeg"))
        print(
            "[astrbot-fact-check-image-split] "
            f"{source_label}: original={len(body)} bytes size={image.size[0]}x{image.size[1]} "
            f"chunks={len(chunks)} chunk_height={chunk_height}",
            flush=True,
        )
        return chunks


def encode_image_chunk_under_limit(image: Image.Image, *, max_bytes: int) -> bytes:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    for quality in (86, 78, 70, 62, 54, 46):
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        body = output.getvalue()
        if len(body) <= max_bytes:
            return body
    output = io.BytesIO()
    scale = max(0.35, (max_bytes / max(1, len(body))) ** 0.5 * 0.9)
    resized = image.resize(
        (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))),
        Image.Resampling.LANCZOS,
    )
    resized.save(output, format="JPEG", quality=50, optimize=True)
    body = output.getvalue()
    if len(body) > max_bytes:
        raise ValueError(f"image chunk too large after compression: {len(body)} bytes > limit {max_bytes}")
    return body


def parse_candidates(text: str, *, limit: int) -> list[ClaimCandidate]:
    text = strip_thinking(text).strip()
    if not text or NO_CHECKABLE_CLAIM in text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().strip('"“”')
        return [ClaimCandidate(cleaned[:1000], "text", 3)] if cleaned else []
    if isinstance(payload, dict):
        payload = payload.get("claims") or payload.get("candidates") or payload.get("questions") or []
    if not isinstance(payload, list):
        return []
    candidates: list[ClaimCandidate] = []
    seen: set[str] = set()
    for item in payload:
        if isinstance(item, str):
            claim = item.strip()
            source = "text"
            priority = 3
        elif isinstance(item, dict):
            claim = str(item.get("question") or item.get("claim") or item.get("text") or "").strip()
            source = str(item.get("source") or "").strip() or "unknown"
            try:
                priority = int(item.get("priority") or 3)
            except (TypeError, ValueError):
                priority = 3
        else:
            continue
        claim = claim.strip().strip('"“”')
        if not claim or NO_CHECKABLE_CLAIM in claim or _is_meta_claim(claim):
            continue
        key = re.sub(r"\s+", "", claim).lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(ClaimCandidate(claim[:1000], source[:120], max(1, min(priority, 5))))
        if len(candidates) >= limit:
            break
    return sorted(candidates, key=lambda x: x.priority, reverse=True)


def _is_meta_claim(claim: str) -> bool:
    """过滤前置模型误抽出的工具说明/核查流程本身。"""
    normalized = re.sub(r"\s+", "", str(claim or "")).strip()
    if not normalized:
        return True
    if META_CLAIM_RE.search(normalized):
        return True
    if normalized in {"请核查：是否属实？", "请核查是否属实？"}:
        return True
    return False


def dedupe_candidates(candidates: list[ClaimCandidate], *, limit: int) -> list[ClaimCandidate]:
    deduped: list[ClaimCandidate] = []
    seen: set[str] = set()
    for item in sorted(candidates, key=lambda x: x.priority, reverse=True):
        keys = _candidate_dedupe_keys(item.claim)
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _candidate_dedupe_keys(claim: str) -> set[str]:
    normalized = normalize_anysearch_query(claim)
    compact = _compact_claim_key(normalized)
    keys = {compact}
    tokens = _claim_tokens(compact)
    if tokens:
        keys.add(" ".join(sorted(tokens)))
    return {key for key in keys if key}


def _compact_claim_key(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"(请核查|是否属实|是否准确|是否真实|是否正确|事实是否准确)", "", text)
    text = re.sub(r"(可以被视为|可被视为|符合|条件|属实|准确|真实|正确)", "", text)
    return re.sub(r"[\s\u00a0\u200b\u200c\u200d:：，,。.!！?？；;、/（）()《》“”\"'\-]+", "", text)


def _claim_tokens(compact: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", compact, flags=re.IGNORECASE))
    stopwords = {"请核查", "是否属实", "是否准确", "可以", "被视为", "符合", "条件"}
    return {token for token in tokens if token not in stopwords}


def format_candidates(candidates: list[ClaimCandidate]) -> str:
    return "\n".join(
        f"{index}. {item.claim}（来自：{item.source or 'unknown'}）"
        for index, item in enumerate(candidates, start=1)
    )


def collect_anysearch_evidence(
    candidates: list[ClaimCandidate],
    *,
    enabled: bool,
    endpoint: str,
    api_key: str,
    timeout: int,
    max_claims: int,
    max_results_per_claim: int,
    extract_top_urls: int,
    max_chars: int,
    freshness: str = "",
    content_types: list[str] | None = None,
) -> AnysearchEvidence:
    if not enabled:
        return AnysearchEvidence()
    endpoint = (endpoint or ANYSEARCH_DEFAULT_ENDPOINT).strip()
    if not endpoint:
        return AnysearchEvidence(reason="disabled: empty endpoint")

    query_payloads = build_anysearch_queries(
        candidates,
        max_claims=max_claims,
        max_results_per_claim=max_results_per_claim,
        freshness=freshness,
        content_types=content_types,
    )
    if not query_payloads:
        return AnysearchEvidence(reason="skipped: no search queries")

    try:
        if len(query_payloads) == 1:
            search_text = anysearch_call_tool(
                tool_name="search",
                arguments=query_payloads[0],
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
            )
        else:
            search_text = anysearch_call_tool(
                tool_name="batch_search",
                arguments={"queries": query_payloads},
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
            )
    except Exception as exc:
        return AnysearchEvidence(reason=f"search failed: {error_label(exc)}")

    urls = [url for url in extract_public_urls(search_text) if is_public_http_url(url)]
    excerpts: list[str] = []
    excerpt_sources: list[str] = []
    for url in urls[: max(0, _clamp_int(extract_top_urls, default=2, lower=0, upper=5))]:
        try:
            extracted = anysearch_call_tool(
                tool_name="extract",
                arguments={"url": url},
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
            )
        except Exception as exc:
            print(
                f"[astrbot-fact-check-anysearch-extract-error] {shorten_text(url, 160)}: {error_label(exc)}",
                flush=True,
            )
            continue
        excerpt_sources.append(url)
        excerpts.append(f"[{len(excerpts) + 1}] {url}\n{shorten_text(extracted, 1800)}")

    sections = [f"搜索摘要：\n{shorten_text(sanitize_anysearch_evidence_text(search_text), 3600)}"]
    if excerpts:
        sections.append("网页正文摘录：\n" + sanitize_anysearch_evidence_text("\n\n".join(excerpts)))
    evidence_text = shorten_text("\n\n".join(sections), _clamp_int(max_chars, default=6000, lower=1000, upper=12000))
    sources = dedupe_sources(excerpt_sources + urls, limit=8)
    return AnysearchEvidence(
        text=evidence_text,
        sources=sources,
        reason=f"ok; queries={len(query_payloads)} urls={len(urls)} extracts={len(excerpts)}",
    )


def build_anysearch_queries(
    candidates: list[ClaimCandidate],
    *,
    max_claims: int,
    max_results_per_claim: int,
    freshness: str = "",
    content_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    max_claims = _clamp_int(max_claims, default=3, lower=1, upper=5)
    max_results = _clamp_int(max_results_per_claim, default=3, lower=1, upper=10)
    if isinstance(content_types, str):
        raw_content_types = [item.strip() for item in content_types.split(",")]
    else:
        raw_content_types = content_types or ["web", "news"]
    normalized_content_types = [
        item
        for item in raw_content_types
        if str(item or "").strip() in ANYSEARCH_CONTENT_TYPES
    ]
    normalized_freshness = freshness if freshness in ANYSEARCH_FRESHNESS_VALUES else ""

    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates[:max_claims]:
        query = normalize_anysearch_query(candidate.claim)
        key = re.sub(r"\s+", "", query).lower()
        if not query or key in seen:
            continue
        seen.add(key)
        payload: dict[str, Any] = {"query": query, "max_results": max_results}
        if normalized_freshness:
            payload["freshness"] = normalized_freshness
        if normalized_content_types:
            payload["content_types"] = normalized_content_types
        queries.append(payload)
    return queries


def normalize_anysearch_query(claim: str) -> str:
    query = str(claim or "").strip()
    query = re.sub(r"^请核查[:：]?\s*", "", query)
    query = re.sub(r"是否属实[？?]?\s*$", "", query)
    query = re.sub(r"\s+", " ", query).strip(" \t\r\n。！？?；;")
    return query[:240]


def anysearch_call_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    endpoint: str,
    api_key: str,
    timeout: int,
    max_retries: int = 1,
) -> str:
    endpoint = (endpoint or ANYSEARCH_DEFAULT_ENDPOINT).strip()
    if not is_public_http_url(endpoint):
        raise ValueError("Anysearch endpoint must be a public http(s) URL")
    clean_api_key = (api_key or os.getenv("ANYSEARCH_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json"}
    if clean_api_key:
        headers["Authorization"] = f"Bearer {clean_api_key}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    http_timeout = httpx.Timeout(
        float(max(3, timeout)),
        connect=min(6.0, float(max(3, timeout))),
        read=float(max(3, timeout)),
        write=6.0,
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=http_timeout, follow_redirects=True, trust_env=True) as client:
                response = client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            if "error" in data:
                error = data["error"]
                if isinstance(error, dict):
                    message = error.get("message") or json.dumps(error, ensure_ascii=False)
                else:
                    message = str(error)
                raise RuntimeError(f"Anysearch API error: {message}")
            return extract_anysearch_text(data)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if not is_retryable(exc) or attempt >= max_retries:
                raise
        except (httpx.RequestError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
        wait = backoff_seconds(attempt, last_error or RuntimeError("Anysearch request failed"))
        print(
            "[astrbot-fact-check-anysearch-retry] "
            f"attempt={attempt + 1}/{max_retries + 1} tool={tool_name} "
            f"wait={wait:.1f}s error={error_label(last_error)}",
            flush=True,
        )
        time.sleep(wait)
    if last_error:
        raise last_error
    raise RuntimeError("Anysearch request failed without a specific error")


def extract_anysearch_text(data: dict[str, Any]) -> str:
    result = data.get("result") or {}
    content = result.get("content") or []
    if isinstance(content, list):
        texts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if texts:
            return "\n".join(text for text in texts if text).strip()
    return json.dumps(result, ensure_ascii=False, indent=2)


def extract_public_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_RE.finditer(str(text or "")):
        url = normalize_url(match.group(1))
        if url and url not in urls:
            urls.append(url)
    return urls


def normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip(".,;:，。；：!?！？)]}>\"'")


def is_public_http_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    if host.lower() in {"localhost", "localhost.localdomain"}:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    )


def dedupe_sources(sources: list[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for source in sources:
        item = str(source or "").strip()
        if not item:
            continue
        key = re.sub(r"\s+", "", item).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def compact_source_label(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return ""
    url_text = text
    if "：" in text:
        _, possible_url = text.rsplit("：", 1)
        if possible_url.strip().lower().startswith(("http://", "https://")):
            url_text = possible_url.strip()
    parsed = urlparse(url_text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        host = parsed.hostname or parsed.netloc
        if host.startswith("www."):
            host = host[4:]
        path_parts = [part for part in parsed.path.strip("/").split("/") if part][:2]
        label = host + ("/" + "/".join(path_parts) if path_parts else "")
        return shorten_text(label, 60).replace("\n", " ")
    return shorten_text(re.sub(r"\s+", " ", text), 60).replace("\n", " ")


def shorten_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    max_chars = max(0, int(max_chars or 0))
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)].rstrip() + "\n...（已截断）"


def _clamp_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(number, upper))


def extract_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates", []) or []
    candidate = candidates[0] if candidates else {}
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    return strip_thinking(text)


def sanitize_anysearch_evidence_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*\*\*URL\*\*\s*[:：]\s*", "URL：", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*[-*]\s*\*\*([^*]+)\*\*\s*[:：]\s*", r"\1：", line)
        line = line.replace("**", "")
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def sanitize_fact_check_reply(text: str) -> str:
    cleaned = sanitize_anysearch_evidence_text(strip_thinking(text))
    cleaned = _split_inline_fact_check_points(cleaned)
    cleaned = _normalize_conditional_verdict(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_inline_fact_check_points(text: str) -> str:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("要点："):
            points_text = stripped.split("：", 1)[1].strip()
            matches = list(re.finditer(r"(\d+)\.\s*", points_text))
            if matches:
                for index, match in enumerate(matches):
                    start = match.end()
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(points_text)
                    point = points_text[start:end].strip(" \t\r\n；;")
                    if point:
                        lines.append(f"{match.group(1)}. 核查点：{point}")
                continue
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        lines.append(stripped)
    return "\n".join(lines)


def _normalize_conditional_verdict(text: str) -> str:
    if not CONDITIONAL_CLAIM_RE.search(text or ""):
        return text
    lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^事实核查[:：]\s*(已证实|大致可信)\b", stripped):
            line = re.sub(r"(事实核查[:：]\s*)(已证实|大致可信)", r"\1条件性成立", line, count=1)
        elif re.match(r"^结论[:：]\s*(已证实|已核实)\b", stripped):
            line = re.sub(r"(结论[:：]\s*)(已证实|已核实)", r"\1条件性成立", line, count=1)
        lines.append(line)
    return "\n".join(lines)


def extract_sources(body: dict[str, Any], limit: int = 3) -> list[str]:
    candidates = body.get("candidates", []) or []
    metadata = (candidates[0] if candidates else {}).get("groundingMetadata") or {}
    chunks = metadata.get("groundingChunks") or []
    sources: list[str] = []
    for chunk in chunks:
        web = (chunk or {}).get("web") or {}
        uri = str(web.get("uri") or "").strip()
        title = str(web.get("title") or "").strip()
        if not uri:
            continue
        item = f"{title or uri}：{uri}"
        if item not in sources:
            sources.append(item)
        if len(sources) >= limit:
            break
    return sources


def request_with_retry(req: request.Request, *, timeout: int, max_retries: int):
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return request.urlopen(req, timeout=timeout)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
                raise
            time.sleep(1.2 * (attempt + 1))
        except URLError as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(1.2 * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("request failed without a specific error")


def guess_mime_type(file_name: str, content_type: str) -> str:
    content_type = (content_type or "").split(";")[0].strip().lower()
    if content_type.startswith("image/"):
        return content_type
    suffix = Path(file_name or "").suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def _normalize(text: str) -> str:
    return str(text or "").replace("\u00a0", " ").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").strip()


def is_retryable(exc: Exception | None) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in RETRYABLE_STATUS_CODES
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, (URLError, httpx.RequestError))


def error_label(exc: Exception | None) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} {exc.response.reason_phrase}"
    if isinstance(exc, httpx.TimeoutException):
        return f"HTTP timeout: {exc}"
    if isinstance(exc, httpx.RequestError):
        return f"HTTP request error: {exc}"
    if isinstance(exc, URLError):
        return f"URL error: {exc.reason}"
    return repr(exc)


def backoff_seconds(attempt: int, exc: Exception) -> float:
    if isinstance(exc, HTTPError) and exc.headers:
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(2.0**attempt, 20.0) + random.uniform(0.1, 0.8)


def post_json_with_timeout(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout: int,
    max_retries: int = 1,
) -> dict[str, Any]:
    http_timeout = httpx.Timeout(float(timeout), connect=min(8.0, float(timeout)), read=float(timeout), write=8.0)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=http_timeout, follow_redirects=True, trust_env=True) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if not is_retryable(exc) or attempt >= max_retries:
                raise
        except httpx.RequestError as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
        wait = backoff_seconds(attempt, last_error or RuntimeError("request failed"))
        print(
            "[astrbot-fact-check-http-retry] "
            f"attempt={attempt + 1}/{max_retries + 1} wait={wait:.1f}s "
            f"error={error_label(last_error)}",
            flush=True,
        )
        time.sleep(wait)
    if last_error:
        raise last_error
    raise RuntimeError("HTTP request failed without a specific error")


def read_image_input_bytes(item: ImageInput, *, max_bytes: int | None, timeout: int) -> tuple[bytes, str]:
    path_value = (item.path or "").strip()
    url_value = (item.url or "").strip()
    if path_value:
        path = Path(path_value.removeprefix("file:///").removeprefix("file://"))
        if not path.exists():
            raise FileNotFoundError(f"local image not found: {path}")
        size = path.stat().st_size
        if max_bytes is not None and size > max_bytes:
            raise ValueError(f"image too large: {size} bytes > limit {max_bytes}")
        return path.read_bytes(), guess_mime_type(item.file_name or path.name, "")
    if url_value.startswith("file://"):
        path = Path(url_value.removeprefix("file:///").removeprefix("file://"))
        if not path.exists():
            raise FileNotFoundError(f"local image not found: {path}")
        size = path.stat().st_size
        if max_bytes is not None and size > max_bytes:
            raise ValueError(f"image too large: {size} bytes > limit {max_bytes}")
        return path.read_bytes(), guess_mime_type(item.file_name or path.name, "")
    if url_value.startswith("base64://"):
        body = base64.b64decode(url_value.removeprefix("base64://"))
        if max_bytes is not None and len(body) > max_bytes:
            raise ValueError(f"image too large: {len(body)} bytes > limit {max_bytes}")
        return body, guess_mime_type(item.file_name, "")
    if url_value:
        return get_bytes_with_timeout(url_value, max_bytes=max_bytes, timeout=timeout)
    raise ValueError("empty image input")


def get_bytes_with_timeout(url: str, *, max_bytes: int | None, timeout: int) -> tuple[bytes, str]:
    http_timeout = httpx.Timeout(float(timeout), connect=min(5.0, float(timeout)), read=float(timeout), write=5.0)
    with httpx.Client(timeout=http_timeout, follow_redirects=True, trust_env=True) as client:
        with client.stream("GET", url, headers={"User-Agent": "AstrBot-QQ-Agent/0.1"}) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length", "")
            if max_bytes is not None and content_length and int(content_length) > max_bytes:
                raise ValueError(f"image too large: {content_length} bytes > limit {max_bytes}")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    raise ValueError(f"image too large while downloading: {size} bytes > limit {max_bytes}")
                chunks.append(chunk)
            return b"".join(chunks), response.headers.get("Content-Type", "")
