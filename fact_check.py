from __future__ import annotations

import base64
import concurrent.futures
import contextvars
import functools
import io
import ipaddress
import json
import os
import random
import re
import socket
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
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
_MODEL_FAILURE_UNTIL: dict[str, float] = {}
_MODEL_FAILURE_LOCK = threading.Lock()
_PUBLIC_HOST_CACHE: dict[str, tuple[float, bool]] = {}
_PUBLIC_HOST_CACHE_LOCK = threading.Lock()
_REQUEST_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "fact_check_request_deadline",
    default=None,
)
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
- 证据必须直接覆盖核心争议命题才可判“可信”或“基本可信但需限定”；只证实外围事实或基础事实不够。
- 对“法规/政策存在 + 具体产品、硬件、功能、销售、违法性推论”的复合命题，必须分别判断：
  1. 法规/政策是否真实存在、发布机构和生效时间是否属实；
  2. 条文、官方解释或权威报道是否直接说明原文中的具体对象/行为被覆盖。
- 若来源只支持法规/政策存在，但没有直接支持具体适用推论，整体结论最高为“部分存疑”，并明确写“未找到直接证据支持该具体推论”。
- 总结论按最关键争议命题决定，不要因为任一子事实成立就把整体判成“可信”或“基本可信但需限定”。
- 要点中尽量标出“已核实 / 条件性成立 / 表述需限定 / 部分存疑 / 证据不足 / 不准确 / 无法判断”。"""
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
    content_sha256: str = ""


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


class IncompleteGenerationError(RuntimeError):
    """A model returned text but did not finish a usable answer."""


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


def _with_request_deadline(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        seconds = float(kwargs.get("total_timeout_seconds") or 0)
        token = _REQUEST_DEADLINE.set(time.monotonic() + seconds if seconds > 0 else None)
        try:
            return func(*args, **kwargs)
        finally:
            _REQUEST_DEADLINE.reset(token)

    return wrapped


def _bounded_timeout(timeout: float, *, minimum: float = 0.25) -> float:
    requested = max(minimum, float(timeout))
    deadline = _REQUEST_DEADLINE.get()
    if deadline is None:
        return requested
    remaining = deadline - time.monotonic()
    if remaining <= minimum:
        raise httpx.TimeoutException("fact-check total deadline exceeded")
    return min(requested, remaining)


def _sleep_with_deadline(seconds: float) -> None:
    delay = max(0.0, float(seconds))
    deadline = _REQUEST_DEADLINE.get()
    if deadline is None:
        time.sleep(delay)
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise httpx.TimeoutException("fact-check total deadline exceeded")
    time.sleep(min(delay, remaining))
    if deadline - time.monotonic() <= 0:
        raise httpx.TimeoutException("fact-check total deadline exceeded")


def _retry_budget_available(*, minimum: float = 0.5) -> bool:
    deadline = _REQUEST_DEADLINE.get()
    return deadline is None or deadline - time.monotonic() > minimum


@_with_request_deadline
def run_fact_check(
    *,
    request_data: FactCheckRequest,
    api_key: str,
    base_url: str,
    pre_model: str,
    evidence_model: str = "",
    verdict_models: list[str] | None = None,
    main_models: list[str] | None = None,
    max_image_bytes: int = 5 * 1024 * 1024,
    long_image_chunk_height: int = 2200,
    long_image_max_parts: int = 8,
    long_image_max_width: int = 1280,
    image_download_timeout: int = 10,
    pre_request_timeout: int = 25,
    main_request_timeout: int = 45,
    evidence_max_output_tokens: int = 1536,
    evidence_retry_max_output_tokens: int = 3072,
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
    model_failure_cooldown_seconds: int = 0,
    verdict_request_timeout: int = 25,
    verdict_max_output_tokens: int = 2048,
    verdict_retry_max_output_tokens: int = 4096,
    source_link_timeout: int = 4,
    total_timeout_seconds: int = 0,
) -> FactCheckResult:
    api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return FactCheckResult(FAILED_REPLY, "missing Gemini API key")

    # `main_models` keeps the callable backwards-compatible for older tests and
    # third-party callers. New configuration uses one grounded evidence model
    # followed by optional evidence-only verdict editors.
    legacy_models = [str(model).strip() for model in (main_models or []) if str(model).strip()]
    evidence_model = str(evidence_model or "").strip() or (legacy_models[0] if legacy_models else "gemini-2.5-flash")
    verdict_models = [str(model).strip() for model in (verdict_models or []) if str(model).strip()]

    candidates: list[ClaimCandidate] = []
    text_context = request_data.text.strip()
    text_preprocess_attempted = False
    main_image_parts = build_inline_image_parts(
        request_data.images,
        max_image_bytes=max_image_bytes,
        long_image_chunk_height=long_image_chunk_height,
        long_image_max_parts=long_image_max_parts,
        long_image_max_width=long_image_max_width,
        image_download_timeout=image_download_timeout,
        stage="shared-reference",
    )
    if text_context and not request_data.images:
        text_preprocess_attempted = True
        print(f"[astrbot-fact-check-stage] text-preprocess start len={len(text_context)} model={pre_model}", flush=True)
        try:
            candidates.extend(
                extract_claims_from_text(
                    text_context,
                    model=pre_model,
                    api_key=api_key,
                    base_url=base_url,
                    request_timeout=pre_request_timeout,
                ),
            )
        except Exception as exc:
            print(
                f"[astrbot-fact-check-text-preprocess-error] {error_label(exc)}",
                flush=True,
            )
        print(f"[astrbot-fact-check-stage] text-preprocess done candidates={len(candidates)}", flush=True)

    if request_data.images:
        print(
            "[astrbot-fact-check-stage] image-preprocess start "
            f"images={len(request_data.images)} model={pre_model}",
            flush=True,
        )
        try:
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
                    inline_parts=main_image_parts,
                ),
            )
        except Exception as exc:
            print(
                f"[astrbot-fact-check-image-preprocess-error] {error_label(exc)}",
                flush=True,
            )
            if text_context:
                text_preprocess_attempted = True
                try:
                    candidates.extend(
                        extract_claims_from_text(
                            text_context,
                            model=pre_model,
                            api_key=api_key,
                            base_url=base_url,
                            request_timeout=pre_request_timeout,
                        ),
                    )
                except Exception as text_exc:
                    print(
                        f"[astrbot-fact-check-text-preprocess-error] {error_label(text_exc)}",
                        flush=True,
                    )
        print(f"[astrbot-fact-check-stage] image-preprocess done candidates={len(candidates)}", flush=True)

    if text_context and not candidates and not text_preprocess_attempted:
        text_preprocess_attempted = True
        try:
            candidates.extend(
                extract_claims_from_text(
                    text_context,
                    model=pre_model,
                    api_key=api_key,
                    base_url=base_url,
                    request_timeout=pre_request_timeout,
                ),
            )
        except Exception as exc:
            print(
                f"[astrbot-fact-check-text-preprocess-error] {error_label(exc)}",
                flush=True,
            )

    if not candidates:
        if text_context:
            candidates.append(
                ClaimCandidate(
                    claim=f"请核查下面聊天内容中涉及的事实是否准确：{text_context[:800]}",
                    source="原始聊天内容",
                    priority=2,
                ),
            )
        elif request_data.images:
            candidates.append(
                ClaimCandidate(
                    claim="请核查图片中主要事实断言是否准确，并指出无法辨认或缺少证据的部分。",
                    source="原始图片",
                    priority=2,
                ),
            )
        else:
            return FactCheckResult(FAILED_REPLY, "no checkable factual claims", [], [])

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
    if main_image_parts:
        image_line = f"参考图片：已附上 {len(main_image_parts)} 张原图。请同时参考图中文字、截图上下文和画面含义。\n"
    elif request_data.images:
        image_line = "参考图片：原图未成功附上，只能依据前置整理出的核查问题和原始文字判断。\n"
    else:
        image_line = ""
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
- 对复合命题逐项写清“已核实 / 条件性成立 / 表述需限定 / 部分存疑 / 证据不足 / 不准确 / 无法判断”，不要只回答其中一个子事实。
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
        "[astrbot-fact-check-stage] evidence-check start "
        f"model={evidence_model} candidates={len(deduped)}",
        flush=True,
    )
    evidence_tokens = _clamp_int(
        evidence_max_output_tokens,
        default=1536,
        lower=512,
        upper=8192,
    )
    evidence_body, evidence_used_model = generate_with_fallback(
        prompt=prompt,
        models=[evidence_model],
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_output_tokens=evidence_tokens,
        grounding=True,
        model_failure_cooldown_seconds=model_failure_cooldown_seconds,
        extra_parts=main_image_parts,
        request_timeout=main_request_timeout,
    )
    try:
        validate_complete_fact_check_result(evidence_body)
    except IncompleteGenerationError as exc:
        retry_tokens = _clamp_int(
            evidence_retry_max_output_tokens,
            default=3072,
            lower=1024,
            upper=8192,
        )
        if not _retry_budget_available(minimum=5.0):
            return FactCheckResult(
                FAILED_REPLY,
                f"grounded evidence incomplete and insufficient retry budget: {exc}",
                candidates=deduped,
            )
        print(
            "[astrbot-fact-check-evidence-retry] "
            f"model={evidence_used_model} max_output_tokens={retry_tokens} reason={exc}",
            flush=True,
        )
        try:
            evidence_body, evidence_used_model = generate_with_fallback(
                prompt=prompt,
                models=[evidence_used_model],
                api_key=api_key,
                base_url=base_url,
                temperature=0.1,
                max_output_tokens=retry_tokens,
                grounding=True,
                model_failure_cooldown_seconds=model_failure_cooldown_seconds,
                extra_parts=main_image_parts,
                request_timeout=main_request_timeout,
            )
            validate_complete_fact_check_result(evidence_body)
        except IncompleteGenerationError as retry_exc:
            print(
                "[astrbot-fact-check-evidence-incomplete] "
                f"model={evidence_used_model} reason={retry_exc}",
                flush=True,
            )
            return FactCheckResult(
                FAILED_REPLY,
                f"grounded evidence incomplete after retry: {retry_exc}",
                candidates=deduped,
            )
    print(
        "[astrbot-fact-check-stage] evidence-check done "
        f"model={evidence_used_model} {generation_diagnostics(evidence_body)}",
        flush=True,
    )

    # Gemini 3 Flash does not need native grounding here. It receives the
    # grounded evidence package from 2.5 Flash and is used only to calibrate
    # atomic-claim verdicts. The grounded result remains a complete fallback.
    body = evidence_body
    used_model = evidence_used_model
    if verdict_models:
        evidence_text = shorten_text(extract_text(evidence_body).strip(), 9000)
        evidence_sources = dedupe_sources(extract_sources(evidence_body) + anysearch_evidence.sources, limit=5)
        grounding_evidence = extract_grounding_evidence(evidence_body)
        verdict_prompt = build_evidence_verdict_prompt(
            request_data=request_data,
            candidates=deduped,
            evidence_text=evidence_text,
            evidence_sources=evidence_sources,
            grounding_evidence=grounding_evidence,
            anysearch_evidence=anysearch_evidence.text,
        )
        print(
            "[astrbot-fact-check-stage] verdict-review start "
            f"models={','.join(verdict_models)}",
            flush=True,
        )
        try:
            verdict_body, verdict_model = generate_with_fallback(
                prompt=verdict_prompt,
                models=verdict_models,
                api_key=api_key,
                base_url=base_url,
                temperature=0.1,
                max_output_tokens=_clamp_int(
                    verdict_max_output_tokens,
                    default=2048,
                    lower=512,
                    upper=8192,
                ),
                grounding=False,
                thinking_level="medium",
                model_failure_cooldown_seconds=model_failure_cooldown_seconds,
                http_max_retries=0,
                request_timeout=verdict_request_timeout,
            )
            try:
                validate_complete_fact_check_result(verdict_body)
            except IncompleteGenerationError as exc:
                retry_tokens = _clamp_int(
                    verdict_retry_max_output_tokens,
                    default=4096,
                    lower=1024,
                    upper=8192,
                )
                print(
                    "[astrbot-fact-check-verdict-retry] "
                    f"model={verdict_model} max_output_tokens={retry_tokens} reason={exc}",
                    flush=True,
                )
                verdict_body, verdict_model = generate_with_fallback(
                    prompt=verdict_prompt,
                    models=[verdict_model],
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.1,
                    max_output_tokens=retry_tokens,
                    grounding=False,
                    thinking_level="medium",
                    model_failure_cooldown_seconds=model_failure_cooldown_seconds,
                    http_max_retries=0,
                    request_timeout=verdict_request_timeout,
                )
                validate_complete_fact_check_result(verdict_body)
            body, used_model = verdict_body, verdict_model
            print(
                "[astrbot-fact-check-stage] verdict-review done "
                f"model={used_model} {generation_diagnostics(verdict_body)}",
                flush=True,
            )
        except Exception as exc:
            print(
                "[astrbot-fact-check-verdict-fallback] "
                f"model={','.join(verdict_models)} error={error_label(exc)}",
                flush=True,
            )
    reply = sanitize_fact_check_reply(extract_text(body).strip())
    sources = normalize_fact_check_sources(dedupe_sources(
        extract_sources(body) + extract_sources(evidence_body) + anysearch_evidence.sources,
        limit=5,
    ), redirect_timeout=source_link_timeout)
    if sources and "来源" not in reply:
        reply += "\n来源：" + "；".join(compact_source_label(source) for source in sources[:3])
    reply = append_source_links(reply, sources)
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


def append_source_links(reply: str, sources: list[str], *, limit: int = 3) -> str:
    urls: list[str] = []
    for source in sources:
        for match in URL_RE.finditer(str(source or "")):
            url = normalize_url(match.group(1))
            if url and is_public_http_url(url) and url not in urls:
                urls.append(url)
            if len(urls) >= limit:
                break
        if len(urls) >= limit:
            break
    missing = [url for url in urls if url not in str(reply or "")]
    if not missing:
        return str(reply or "").strip()
    links = "\n".join(f"{index}. {url}" for index, url in enumerate(missing, start=1))
    return f"{str(reply or '').rstrip()}\n可核验链接：\n{links}".strip()


def build_evidence_verdict_prompt(
    *,
    request_data: FactCheckRequest,
    candidates: list[ClaimCandidate],
    evidence_text: str,
    evidence_sources: list[str],
    grounding_evidence: str = "",
    anysearch_evidence: str = "",
) -> str:
    """Build the non-grounded Gemini 3 verdict pass from a grounded evidence package."""
    sources = "\n".join(f"- {source}" for source in evidence_sources[:5]) or "- No source URL was returned."
    return f"""You are the final Chinese fact-check verdict editor.

Use only the grounded evidence package below. Do not browse, invent sources, or fill gaps with prior knowledge.
The evidence model may have verified a background fact without proving a later legal, medical, financial, policy, product, hardware, sales, or illegality inference. Split such compound claims into atomic claims. If a key inference lacks direct support, the overall conclusion must not be trustworthy; use a cautious partial/insufficient verdict instead.

Current time:
{current_time_context()}

Original chat content:
{request_data.text or '(image-led request)'}

Checkable claims:
{format_candidates(candidates)}

Grounded evidence package:
{evidence_text or '(The grounded model returned no readable text.)'}

Grounding support mapping from Google Search:
{grounding_evidence or '(No grounding support mapping was returned.)'}

Raw Anysearch excerpts and search snippets:
{sanitize_anysearch_evidence_text(anysearch_evidence) or '(No Anysearch evidence was returned.)'}

Grounded source URLs:
{sources}

Write a concise Chinese QQ-ready result. Start with "事实核查：" and choose one overall verdict from: 可信 / 基本可信但需限定 / 条件性成立 / 混合结论 / 部分存疑 / 证据不足 / 基本不实 / 表述不准确.
For every atomic claim write: "结论：" followed by one of: 已核实 / 条件性成立 / 表述需限定 / 部分存疑 / 证据不足 / 不准确 / 无法判断.
State clearly when the evidence does not directly support a key inference. End with at most three actual source domains or titles. Do not use Markdown headings, bold, code blocks, or fabricated citations."""


def extract_grounding_evidence(body: dict[str, Any], *, max_chars: int = 6000) -> str:
    candidates = body.get("candidates", []) or []
    metadata = (candidates[0] if candidates else {}).get("groundingMetadata") or {}
    chunks = metadata.get("groundingChunks") or []
    supports = metadata.get("groundingSupports") or []
    lines: list[str] = []
    for support in supports:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment") or {}
        segment_text = shorten_text(str(segment.get("text") or "").strip(), 500)
        source_labels: list[str] = []
        for index in support.get("groundingChunkIndices") or []:
            if not isinstance(index, int) or index < 0 or index >= len(chunks):
                continue
            web = (chunks[index] or {}).get("web") or {}
            uri = str(web.get("uri") or "").strip()
            title = str(web.get("title") or "").strip()
            if uri:
                source_labels.append(f"{title or compact_source_label(uri)}：{uri}")
        if segment_text or source_labels:
            line = f"证据片段：{segment_text or '（未返回原文片段）'}"
            if source_labels:
                line += "\n直接支持来源：" + "；".join(source_labels[:3])
            lines.append(line)
    if not lines:
        for chunk in chunks[:5]:
            web = (chunk or {}).get("web") or {}
            uri = str(web.get("uri") or "").strip()
            title = str(web.get("title") or "").strip()
            if uri:
                lines.append(f"检索来源：{title or compact_source_label(uri)}：{uri}")
    return shorten_text("\n\n".join(lines), max_chars)


@_with_request_deadline
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
    source_link_timeout: int = 4,
    total_timeout_seconds: int = 0,
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
    sources = normalize_fact_check_sources(
        extract_sources(body),
        redirect_timeout=source_link_timeout,
    )
    if sources and "来源" not in reply:
        reply += "\n来源：" + "；".join(compact_source_label(source) for source in sources)
    reply = append_source_links(reply, sources)
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
        thinking_level="minimal",
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
    inline_parts: list[dict[str, Any]] | None = None,
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
    if inline_parts is not None:
        parts.extend(inline_parts)
    else:
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
        thinking_level="minimal",
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
    ungrounded_models: list[str] | None = None,
    model_failure_cooldown_seconds: int = 0,
    extra_parts: list[dict[str, Any]] | None = None,
    max_attempts: int | None = None,
    http_max_retries: int = 1,
    request_timeout: int = 45,
    thinking_level: str | None = None,
) -> tuple[dict[str, Any], str]:
    clean_models = [model.strip() for model in models if str(model or "").strip()]
    if not clean_models:
        raise RuntimeError("no fact-check model configured")
    active_models = _available_models(clean_models)
    attempts = max_attempts or max(1, min(3, len(active_models)))
    last_error: Exception | None = None
    last_model = active_models[0]
    ungrounded = {str(model).strip() for model in (ungrounded_models or []) if str(model).strip()}
    for attempt in range(attempts):
        model = active_models[min(attempt, len(active_models) - 1)]
        last_model = model
        model_grounding = grounding and model not in ungrounded
        try:
            return gemini_generate(
                prompt=prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                grounding=model_grounding,
                extra_parts=extra_parts,
                http_max_retries=http_max_retries,
                request_timeout=request_timeout,
                thinking_level=thinking_level,
            ), model
        except Exception as exc:
            last_error = exc
            _mark_model_unavailable(
                model,
                exc,
                cooldown_seconds=model_failure_cooldown_seconds,
            )
            if not is_retryable(exc) or attempt >= attempts - 1 or not _retry_budget_available():
                break
            next_model = active_models[min(attempt + 1, len(active_models) - 1)]
            wait = backoff_seconds(attempt, exc)
            print(
                "[astrbot-fact-check-retry] "
                f"attempt={attempt + 1}/{attempts} model={model} next={next_model} "
                f"grounding={'on' if model_grounding else 'off'} "
                f"wait={wait:.1f}s error={error_label(exc)}"
                ,
                flush=True,
            )
            _sleep_with_deadline(wait)
    raise RuntimeError(
        f"fact-check request failed after {attempts} attempt(s); "
        f"last_model={last_model}; last_error={error_label(last_error)}"
    )


def _available_models(models: list[str]) -> list[str]:
    now = time.monotonic()
    with _MODEL_FAILURE_LOCK:
        for model, until in list(_MODEL_FAILURE_UNTIL.items()):
            if until <= now:
                _MODEL_FAILURE_UNTIL.pop(model, None)
        active = [model for model in models if _MODEL_FAILURE_UNTIL.get(model, 0.0) <= now]
    return active or models


def _mark_model_unavailable(model: str, exc: Exception, *, cooldown_seconds: int) -> None:
    if cooldown_seconds <= 0 or not _is_model_capacity_error(exc):
        return
    seconds = max(1, int(cooldown_seconds))
    with _MODEL_FAILURE_LOCK:
        _MODEL_FAILURE_UNTIL[model] = time.monotonic() + seconds
    print(
        "[astrbot-fact-check-model-cooldown] "
        f"model={model} seconds={seconds} error={error_label(exc)}",
        flush=True,
    )


def _is_model_capacity_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 503}
    lowered = str(exc or "").lower()
    return "429" in lowered or "503" in lowered or "too many requests" in lowered


def build_generation_config(
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
    if model.startswith("gemini-3"):
        if thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": thinking_level}
        return config
    config["temperature"] = temperature
    if model.startswith("gemini-2.5-flash"):
        config["thinkingConfig"] = {"thinkingBudget": 0}
    return config


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
    http_max_retries: int = 1,
    request_timeout: int = 45,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    generation_config = build_generation_config(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
    )
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
        max_retries=max(0, int(http_max_retries)),
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
    thinking_level: str | None = None,
) -> str:
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": build_generation_config(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        ),
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
    extract_urls = urls[: max(0, _clamp_int(extract_top_urls, default=2, lower=0, upper=5))]
    request_deadline = _REQUEST_DEADLINE.get()

    def extract_page(url: str) -> tuple[str, str, str]:
        token = _REQUEST_DEADLINE.set(request_deadline)
        try:
            ensure_public_url_target(url)
            extracted = anysearch_call_tool(
                tool_name="extract",
                arguments={"url": url},
                endpoint=endpoint,
                api_key=api_key,
                timeout=min(max(3, timeout), 10),
                max_retries=0,
            )
        except Exception as exc:
            return url, "", error_label(exc)
        finally:
            _REQUEST_DEADLINE.reset(token)
        return url, extracted, ""

    extract_results: list[tuple[str, str, str]] = []
    if extract_urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, len(extract_urls))) as executor:
            extract_results = list(executor.map(extract_page, extract_urls))

    excerpts: list[str] = []
    excerpt_sources: list[str] = []
    for url, extracted, error in extract_results:
        if error:
            print(
                f"[astrbot-fact-check-anysearch-extract-error] {shorten_text(url, 160)}: {error}",
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
    ensure_public_url_target(endpoint)
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
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            bounded = _bounded_timeout(max(3, timeout))
            http_timeout = httpx.Timeout(
                bounded,
                connect=min(6.0, bounded),
                read=bounded,
                write=min(6.0, bounded),
            )
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
            if not is_retryable(exc) or attempt >= max_retries or not _retry_budget_available():
                raise
        except (httpx.RequestError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt >= max_retries or not _retry_budget_available():
                raise
        wait = backoff_seconds(attempt, last_error or RuntimeError("Anysearch request failed"))
        print(
            "[astrbot-fact-check-anysearch-retry] "
            f"attempt={attempt + 1}/{max_retries + 1} tool={tool_name} "
            f"wait={wait:.1f}s error={error_label(last_error)}",
            flush=True,
        )
        _sleep_with_deadline(wait)
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
    return ip.is_global


def ensure_public_url_target(url: str) -> None:
    if not is_public_http_url(url):
        raise ValueError("URL must use public http(s)")
    host = str(urlparse(url).hostname or "").lower()
    try:
        ipaddress.ip_address(host)
        return
    except ValueError:
        pass
    now = time.monotonic()
    with _PUBLIC_HOST_CACHE_LOCK:
        cached = _PUBLIC_HOST_CACHE.get(host)
        if cached and now - cached[0] < 60:
            if not cached[1]:
                raise ValueError("URL hostname resolves to a non-public address")
            return
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            if item and item[4]
        }
    except OSError as exc:
        raise ValueError(f"URL hostname could not be resolved: {host}") from exc
    public = bool(addresses) and all(ipaddress.ip_address(address).is_global for address in addresses)
    with _PUBLIC_HOST_CACHE_LOCK:
        _PUBLIC_HOST_CACHE[host] = (now, public)
    if not public:
        raise ValueError("URL hostname resolves to a non-public address")


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


def normalize_fact_check_sources(sources: list[str], *, redirect_timeout: int = 4) -> list[str]:
    budget_seconds = _clamp_int(redirect_timeout, default=4, lower=1, upper=10)
    started_at = time.monotonic()
    request_deadline = _REQUEST_DEADLINE.get()
    deadline = started_at + budget_seconds
    if request_deadline is not None:
        deadline = min(deadline, request_deadline)
    normalized: list[str] = []
    resolved_count = 0
    unresolved_count = 0
    for source in sources[:3]:
        title, url = split_source_title_url(source)
        if is_google_grounding_redirect(url):
            resolved = resolve_google_grounding_redirect(
                url,
                timeout=budget_seconds,
                deadline=deadline,
            )
            if resolved:
                resolved_count += 1
            else:
                unresolved_count += 1
            source = f"{title}：{resolved}" if resolved and title else (resolved or title or "Google Search 来源")
        elif url:
            source = f"{title}：{url}" if title else url
        else:
            source = title
        if source and source not in normalized:
            normalized.append(source)
    elapsed = time.monotonic() - started_at
    if resolved_count or unresolved_count:
        print(
            "[astrbot-fact-check-source-links] "
            f"resolved={resolved_count} unresolved={unresolved_count} elapsed={elapsed:.2f}s "
            f"budget={budget_seconds}s",
            flush=True,
        )
    return normalized


def split_source_title_url(source: str) -> tuple[str, str]:
    text = str(source or "").strip()
    match = URL_RE.search(text)
    if not match:
        return text, ""
    url = normalize_url(match.group(1))
    return text[: match.start()].rstrip(" ：:"), url


def is_google_grounding_redirect(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    return (
        (parsed.hostname or "").lower() == "vertexaisearch.cloud.google.com"
        and parsed.path.startswith("/grounding-api-redirect/")
    )


def resolve_google_grounding_redirect(url: str, *, timeout: int, deadline: float | None = None) -> str:
    current = normalize_url(url)
    timeout = _clamp_int(timeout, default=4, lower=1, upper=10)
    deadline = deadline if deadline is not None else time.monotonic() + timeout
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AstrBotFactCheck/1.0)"}
    try:
        with httpx.Client(follow_redirects=False, headers=headers) as client:
            for _ in range(3):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ""
                ensure_public_url_target(current)
                with client.stream("GET", current, timeout=min(2.0, remaining)) as response:
                    location = response.headers.get("location")
                    if response.status_code not in {301, 302, 303, 307, 308} or not location:
                        return (
                            current
                            if response.is_success
                            and is_public_http_url(current)
                            and not is_google_grounding_redirect(current)
                            else ""
                        )
                current = normalize_url(urljoin(current, location))
                if not current:
                    return ""
    except (httpx.HTTPError, OSError, ValueError):
        return ""
    return ""


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


def generation_diagnostics(body: dict[str, Any]) -> str:
    candidates = body.get("candidates", []) or []
    candidate = candidates[0] if candidates else {}
    usage = body.get("usageMetadata") or {}
    finish = str(candidate.get("finishReason") or "unspecified")
    output_tokens = usage.get("candidatesTokenCount", "?")
    thought_tokens = usage.get("thoughtsTokenCount", "?")
    return f"finish={finish} output_tokens={output_tokens} thought_tokens={thought_tokens}"


def validate_complete_fact_check_result(body: dict[str, Any]) -> None:
    candidates = body.get("candidates", []) or []
    candidate = candidates[0] if candidates else {}
    finish_reason = str(candidate.get("finishReason") or "").strip().upper()
    if finish_reason != "STOP":
        raise IncompleteGenerationError(f"finish_reason={finish_reason}")
    text = sanitize_fact_check_reply(extract_text(body).strip())
    if not text:
        raise IncompleteGenerationError("no readable text")
    if not re.search(r"(?:^|\n)事实核查[：:]\s*\S+", text):
        raise IncompleteGenerationError("missing fact-check summary")
    if not re.search(r"(?:^|\n)结论[：:]\s*\S+", text):
        raise IncompleteGenerationError("missing claim verdict")
    if text.rstrip().endswith((",", "，", "、", ":", "：", ";", "；", "/", "（")):
        raise IncompleteGenerationError("reply ended mid-sentence")


def validate_complete_verdict(body: dict[str, Any]) -> None:
    """Backward-compatible name for callers that validate a final verdict."""
    validate_complete_fact_check_result(body)


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
    cleaned = _break_inline_fact_check_labels(cleaned)
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
                        lines.extend(_format_inline_fact_check_point(match.group(1), point))
                continue
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        lines.append(stripped)
    return "\n".join(lines)


def _format_inline_fact_check_point(number: str, point: str) -> list[str]:
    point = str(point or "").strip()
    verdict = re.match(
        r"^(已证实|已核实|未直接证实|存疑|部分存疑|证据不足|不准确|基本不实|无法判断|条件性成立|表述需限定)[:：]\s*(.+)$",
        point,
    )
    if not verdict:
        return [f"{number}. 核查点：{point}"]
    label = _normalize_verdict_label(verdict.group(1), verdict.group(2))
    return [f"{number}. 核查点：{verdict.group(2).strip()}", f"结论：{label}"]


def _normalize_verdict_label(label: str, context: str = "") -> str:
    label = str(label or "").strip()
    if label in {"已证实", "已核实"}:
        return "条件性成立" if CONDITIONAL_CLAIM_RE.search(context or "") else "已核实"
    if label == "未直接证实":
        return "证据不足"
    if label == "存疑":
        return "部分存疑"
    if label == "基本不实":
        return "不准确"
    return label


def _break_inline_fact_check_labels(text: str) -> str:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        line = re.sub(
            r"\s+(结论|依据|来源|补充依据|是否改变原结论)[:：]",
            lambda match: "\n" + match.group(1) + "：",
            line,
        )
        lines.extend(part.strip() for part in line.splitlines())
    return "\n".join(lines)


def _normalize_conditional_verdict(text: str) -> str:
    if not CONDITIONAL_CLAIM_RE.search(text or ""):
        return text
    output: list[str] = []
    block: list[str] = []

    def normalize_line(line: str, *, conditional: bool) -> str:
        if not conditional:
            return line
        stripped = line.strip()
        if re.match(r"^事实核查[:：]\s*(已证实|已核实|大致可信|基本可信)(?=[\s，,。；;]|$)", stripped):
            return re.sub(
                r"(事实核查[:：]\s*)(已证实|已核实|大致可信|基本可信)",
                r"\1条件性成立",
                line,
                count=1,
            )
        if re.match(r"^结论[:：]\s*(已证实|已核实|可信|大致可信)(?=[\s，,。；;]|$)", stripped):
            return re.sub(
                r"(结论[:：]\s*)(已证实|已核实|可信|大致可信)",
                r"\1条件性成立",
                line,
                count=1,
            )
        return line

    def flush_block() -> None:
        if not block:
            return
        conditional = bool(CONDITIONAL_CLAIM_RE.search("\n".join(block)))
        output.extend(normalize_line(line, conditional=conditional) for line in block)
        block.clear()

    text_has_condition = bool(CONDITIONAL_CLAIM_RE.search(text or ""))
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^\d+\.\s*核查点[:：]", stripped):
            flush_block()
            block.append(line)
            continue
        if block:
            block.append(line)
            continue
        output.append(normalize_line(line, conditional=text_has_condition))
    flush_block()
    return "\n".join(output)


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
            return request.urlopen(req, timeout=_bounded_timeout(timeout))
        except HTTPError as exc:
            last_error = exc
            if (
                exc.code not in RETRYABLE_STATUS_CODES
                or attempt >= max_retries
                or not _retry_budget_available()
            ):
                raise
            _sleep_with_deadline(1.2 * (attempt + 1))
        except URLError as exc:
            last_error = exc
            if attempt >= max_retries or not _retry_budget_available():
                raise
            _sleep_with_deadline(1.2 * (attempt + 1))
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
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            bounded = _bounded_timeout(timeout)
            http_timeout = httpx.Timeout(
                bounded,
                connect=min(8.0, bounded),
                read=bounded,
                write=min(8.0, bounded),
            )
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
            if not is_retryable(exc) or attempt >= max_retries or not _retry_budget_available():
                raise
        except httpx.RequestError as exc:
            last_error = exc
            if attempt >= max_retries or not _retry_budget_available():
                raise
        wait = backoff_seconds(attempt, last_error or RuntimeError("request failed"))
        print(
            "[astrbot-fact-check-http-retry] "
            f"attempt={attempt + 1}/{max_retries + 1} wait={wait:.1f}s "
            f"error={error_label(last_error)}",
            flush=True,
        )
        _sleep_with_deadline(wait)
    if last_error:
        raise last_error
    raise RuntimeError("HTTP request failed without a specific error")


def read_image_input_bytes(item: ImageInput, *, max_bytes: int | None, timeout: int) -> tuple[bytes, str]:
    path_value = (item.path or "").strip()
    url_value = (item.url or "").strip()
    if path_value:
        path = Path(path_value.removeprefix("file:///").removeprefix("file://"))
        if path.exists():
            size = path.stat().st_size
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"image too large: {size} bytes > limit {max_bytes}")
            return path.read_bytes(), guess_mime_type(item.file_name or path.name, "")
        if not url_value:
            raise FileNotFoundError(f"local image not found: {path}")
    if url_value:
        if not is_public_http_url(url_value):
            raise ValueError("image url must be public http(s)")
        return get_bytes_with_timeout(url_value, max_bytes=max_bytes, timeout=timeout)
    raise ValueError("empty image input")


def get_bytes_with_timeout(url: str, *, max_bytes: int | None, timeout: int) -> tuple[bytes, str]:
    if not is_public_http_url(url):
        raise ValueError("image url must be public http(s)")
    bounded = _bounded_timeout(timeout)
    http_timeout = httpx.Timeout(
        bounded,
        connect=min(5.0, bounded),
        read=bounded,
        write=min(5.0, bounded),
    )
    current_url = url
    ensure_public_url_target(current_url)
    with httpx.Client(timeout=http_timeout, follow_redirects=False, trust_env=True) as client:
        for _ in range(5):
            if not is_public_http_url(current_url):
                raise ValueError("image redirect target must be public http(s)")
            ensure_public_url_target(current_url)
            response_cm = client.stream("GET", current_url, headers={"User-Agent": "AstrBot-QQ-Agent/0.1"})
            with response_cm as response:
                if 300 <= response.status_code < 400 and response.headers.get("Location"):
                    current_url = urljoin(str(response.url), response.headers["Location"])
                    continue
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
        raise ValueError("too many image redirects")
