from __future__ import annotations

import asyncio
import difflib
import hashlib
import html
import json
import re
import shutil
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Forward, Image, Node, Nodes, Plain, Reply
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.custom_filter import CustomFilter
from astrbot.core.utils.quoted_message.chain_parser import OneBotPayloadParser
from astrbot.core.utils.quoted_message.extractor import (
    extract_quoted_message_images,
    extract_quoted_message_text,
)
from astrbot.core.utils.quoted_message.onebot_client import OneBotClient

from .fact_check import (
    FAILED_REPLY,
    ClaimCandidate,
    FactCheckRequest,
    FactCheckResult,
    ImageInput,
    ensure_claim_points_visible,
    explain_failure,
    infer_anysearch_freshness,
    is_public_http_url,
    is_trigger,
    remove_trigger,
    run_fact_check,
    run_fact_check_followup,
    safe_image_log_label,
)
from .pipeline_config import build_fact_check_kwargs
from .runtime import AsyncSingleFlight, run_blocking_with_timeout
from .storage import FactCheckMetricsStore, atomic_write_json, read_json_file

FACT_CHECK_PIPELINE_VERSION = "quality-v6"

try:
    from astrbot_plugin_access_control.access_control import is_plugin_allowed
except Exception:
    try:
        plugins_dir = Path(__file__).resolve().parents[1]
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        from astrbot_plugin_access_control.access_control import is_plugin_allowed
    except Exception:
        is_plugin_allowed = None

try:
    from astrbot_plugin_qq_agent_core.media_send import (
        is_confirm_timeout,
        send_chain_result,
    )
except Exception:
    try:
        plugins_dir = Path(__file__).resolve().parents[1]
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        from astrbot_plugin_qq_agent_core.media_send import (
            is_confirm_timeout,
            send_chain_result,
        )
    except Exception:
        is_confirm_timeout = None
        send_chain_result = None


def _event_text_candidates(event: AstrMessageEvent) -> list[str]:
    candidates: list[str] = []
    for value in (
        getattr(event, "message_str", ""),
        event.get_message_str() if hasattr(event, "get_message_str") else "",
    ):
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    try:
        for comp in event.get_messages():
            if isinstance(comp, Plain):
                text = str(getattr(comp, "text", "") or "").strip()
                if text and text not in candidates:
                    candidates.append(text)
    except Exception:
        pass
    return candidates


def _trigger_text(event: AstrMessageEvent) -> str:
    for text in _event_text_candidates(event):
        if is_trigger(text):
            return text
    return ""


class FactCheckWakeFilter(CustomFilter):
    """Wake only for explicit fact-check triggers."""

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        return bool(_trigger_text(event))


@dataclass(slots=True)
class FactCheckSession:
    session_id: str
    created_at: float
    group_id: str
    user_id: str
    request_data: FactCheckRequest
    reply: str
    updated_at: float = 0.0
    candidates: list = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class FactCheckPlugin(Star):
    """Standalone QQ-friendly fact-check command."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._fact_check_semaphore = asyncio.Semaphore(
            max(1, int(self.config.get("fact_check_max_concurrent") or 1)),
        )
        self._fact_check_tasks: set[asyncio.Task] = set()
        self._followup_tasks: set[asyncio.Task] = set()
        self._active_followup_jobs = 0
        self._reply_cache: dict[str, tuple[float, float, FactCheckResult]] = {}
        self._singleflight: AsyncSingleFlight[FactCheckResult] = AsyncSingleFlight()
        self._fact_check_sessions: dict[str, FactCheckSession] = {}
        self._metrics_store = FactCheckMetricsStore(
            Path(StarTools.get_data_dir()) / "fact_check_metrics.json",
        )
        self._cooldown_until = 0.0
        self._session_store_enabled = bool(
            self.config.get("fact_check_session_store_enabled", True),
        )
        self._load_fact_check_sessions()
        self._cleanup_forward_failure_dump()

    @filter.event_message_type(filter.EventMessageType.ALL, priority=998_500)
    async def fact_check_followup(self, event: AstrMessageEvent):
        """Answer follow-up questions by replying to a previous fact-check result."""
        if not bool(self.config.get("enable_fact_check", True)):
            return
        if _trigger_text(event):
            return
        question = self._extract_followup_question(event)
        if not question:
            return
        session, missing_context = await self._find_followup_session_with_state(event)
        if not session:
            if missing_context and self._is_fact_check_allowed(event):
                event.set_extra("qq_agent_command_handled", True)
                event.stop_event()
                await event.send(
                    event.plain_result(
                        "这条事实核查上下文已过期，请重新发送 /事实核查 再查一次。"
                    )
                )
            return
        if not self._session_visible_to_event(session, event):
            return
        if not self._is_fact_check_allowed(event):
            return

        event.set_extra("qq_agent_command_handled", True)
        event.stop_event()
        label = self._event_label(event)
        started_at = time.perf_counter()
        cooldown_left = self._cooldown_left()
        if cooldown_left > 0:
            logger.warning(
                f"[astrbot-fact-check-followup-cooldown] {label}: "
                f"left={cooldown_left:.1f}s session={session.session_id}",
            )
            await event.send(
                event.plain_result(
                    f"核查模型暂时繁忙，先冷却 {int(cooldown_left) + 1} 秒。",
                ),
            )
            return
        if self._fact_check_queue_full():
            logger.warning(
                f"[astrbot-fact-check-followup-queue-full] {label}: "
                f"jobs={self._active_fact_check_jobs()} max={self._max_fact_check_queue()}",
            )
            await event.send(
                event.plain_result("事实核查队列满了，等前面的跑完再试一下。")
            )
            return

        followup_task = asyncio.current_task()
        if followup_task is not None:
            self._followup_tasks.add(followup_task)
            followup_task.add_done_callback(self._followup_tasks.discard)
        await event.send(event.plain_result("我接着查一下。"))
        self._active_followup_jobs = (
            max(0, int(getattr(self, "_active_followup_jobs", 0))) + 1
        )
        try:
            total_timeout = max(
                10,
                int(self.config.get("fact_check_total_timeout_seconds") or 90),
            )
            result = await run_blocking_with_timeout(
                partial(
                        run_fact_check_followup,
                        original_text=session.request_data.text,
                        candidates=session.candidates,
                        previous_reply=session.reply,
                        previous_sources=session.sources,
                        question=question,
                        api_key=str(self.config.get("gemini_api_key") or ""),
                        base_url=str(
                            self.config.get(
                                "gemini_base_url",
                                "https://generativelanguage.googleapis.com/v1beta/models",
                            )
                            or "https://generativelanguage.googleapis.com/v1beta/models"
                        ),
                        main_models=[
                            str(
                                self.config.get("fact_check_evidence_model")
                                or "gemini-2.5-flash"
                            ).strip(),
                        ],
                        request_timeout=int(
                            self.config.get("fact_check_main_timeout_seconds") or 45
                        ),
                        max_output_tokens=int(
                            self.config.get("fact_check_followup_max_output_tokens")
                            or 1024,
                        ),
                        retry_max_output_tokens=int(
                            self.config.get(
                                "fact_check_followup_retry_max_output_tokens"
                            )
                            or 2048,
                        ),
                        total_timeout_seconds=total_timeout,
                ),
                timeout=total_timeout,
                capacity=self._fact_check_semaphore,
            )
        except asyncio.TimeoutError:
            reason = f"follow-up timeout after {time.perf_counter() - started_at:.1f}s"
            self._record_fact_check_metric(
                success=False,
                elapsed=time.perf_counter() - started_at,
                followup=True,
                failure_stage="followup_timeout",
            )
            logger.error(f"[astrbot-fact-check-followup-error] {label}: {reason}")
            await self._send_fact_check_reply(
                event,
                self._failed_fact_check_reply(reason),
                label=label,
                purpose="followup-timeout",
            )
            return
        except Exception as exc:
            reason = f"follow-up exception: {exc!r}"
            self._record_fact_check_metric(
                success=False,
                elapsed=time.perf_counter() - started_at,
                followup=True,
                failure_stage="followup_exception",
            )
            logger.error(f"[astrbot-fact-check-followup-error] {label}: {exc!r}")
            self._maybe_start_cooldown(reason)
            await self._send_fact_check_reply(
                event,
                self._failed_fact_check_reply(reason),
                label=label,
                purpose="followup-exception",
            )
            return
        finally:
            self._active_followup_jobs = max(
                0, int(getattr(self, "_active_followup_jobs", 0)) - 1
            )
            if followup_task is not None:
                self._followup_tasks.discard(followup_task)

        logger.info(
            f"[astrbot-fact-check-followup-done] {label}: "
            f"session={session.session_id} {time.perf_counter() - started_at:.2f}s",
        )
        self._record_fact_check_metric(
            success=self._is_successful_result(result),
            elapsed=time.perf_counter() - started_at,
            followup=True,
            partial=str(result.reason or "").startswith("ok; partial"),
        )
        sent = await self._send_fact_check_reply(
            event,
            result.reply or FAILED_REPLY,
            label=label,
            purpose="followup",
            session_id=session.session_id,
        )
        if sent and self._is_successful_result(result):
            session.reply = result.reply or session.reply
            session.sources = self._merge_sources(
                result.sources,
                session.sources,
                limit=5,
            )
            session.updated_at = time.time()
            self._persist_fact_check_sessions()

    @filter.custom_filter(FactCheckWakeFilter, priority=998_000)
    async def fact_check(self, event: AstrMessageEvent):
        """Run fact-checking when users say /事实核查, factcheck, or fact-check."""
        if not bool(self.config.get("enable_fact_check", True)):
            return
        trigger_text = _trigger_text(event)
        if not trigger_text:
            return

        started_at = time.perf_counter()
        event.set_extra("qq_agent_command_handled", True)
        event.stop_event()
        if not self._is_fact_check_allowed(event):
            yield event.plain_result("这个群没开事实核查。")
            return

        request_data = await self._build_fact_check_request(
            event, trigger_text=trigger_text
        )
        if not request_data.text and not request_data.images:
            if self._is_fact_check_command_only(trigger_text):
                yield event.plain_result(self._fact_check_usage_text())
                return
            reason = "no quoted text or inline claim"
            logger.info(
                f"[astrbot-fact-check-reason] {self._event_label(event)}: {reason}"
            )
            yield event.plain_result(self._failed_fact_check_reply(reason))
            return

        cache_key = self._request_cache_key(request_data)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(
                f"[astrbot-fact-check-cache-hit] {self._event_label(event)}: key={cache_key[:12]}"
            )
            cached_result.reply = ensure_claim_points_visible(
                cached_result.reply,
                cached_result.candidates,
            )
            session_id = self._remember_fact_check_session(
                event, request_data, cached_result
            )
            self._record_fact_check_metric(
                success=True,
                elapsed=time.perf_counter() - started_at,
                cache_hit=True,
                partial=str(cached_result.reason or "").startswith("ok; partial"),
            )
            await self._send_fact_check_reply(
                event,
                cached_result.reply or FAILED_REPLY,
                label=self._event_label(event),
                purpose="cache",
                session_id=session_id,
            )
            return

        cooldown_left = self._cooldown_left()
        if cooldown_left > 0:
            logger.warning(
                f"[astrbot-fact-check-cooldown] {self._event_label(event)}: "
                f"left={cooldown_left:.1f}s key={cache_key[:12]}",
            )
            yield event.plain_result(
                f"核查模型暂时繁忙，先冷却 {int(cooldown_left) + 1} 秒。",
            )
            return

        timeout_seconds = max(
            10.0,
            float(self.config.get("fact_check_total_timeout_seconds") or 90),
        )

        async def compute() -> FactCheckResult:
            return await run_blocking_with_timeout(
                partial(self._run_fact_check_sync, request_data, timeout_seconds),
                timeout=timeout_seconds,
                capacity=self._fact_check_semaphore,
            )

        if self._fact_check_queue_full() and not self._singleflight.has(cache_key):
            logger.warning(
                f"[astrbot-fact-check-queue-full] {self._event_label(event)}: "
                f"jobs={self._active_fact_check_jobs()} max={self._max_fact_check_queue()}",
            )
            yield event.plain_result("事实核查队列满了，等前面的跑完再试一下。")
            return

        joining_hint = self._singleflight.has(cache_key)
        try:
            await event.send(
                event.plain_result(
                    "同一条正在核查，我复用结果。" if joining_hint else "我先查一下。"
                )
            )
        except Exception as exc:
            logger.warning(
                f"[astrbot-fact-check-progress-send-failed] "
                f"{self._event_label(event)}: {exc!r}",
            )
            return

        pipeline_task, joining_existing = self._singleflight.start_if(
            cache_key,
            compute,
            can_start=lambda: not self._fact_check_queue_full(),
        )
        if pipeline_task is None:
            logger.warning(
                f"[astrbot-fact-check-queue-full] {self._event_label(event)}: "
                f"jobs={self._active_fact_check_jobs()} max={self._max_fact_check_queue()}",
            )
            yield event.plain_result("事实核查队列满了，等前面的跑完再试一下。")
            return

        logger.info(
            f"[astrbot-fact-check-queue] {self._event_label(event)}: "
            f"text_len={len(request_data.text)} images={len(request_data.images)} "
            f"active={len(self._fact_check_tasks)} key={cache_key[:12]}",
        )
        task = asyncio.create_task(
            self._run_fact_check_job(
                event,
                request_data,
                started_at,
                cache_key,
                pipeline_task=pipeline_task,
                joined_existing=joining_existing,
            )
        )
        self._fact_check_tasks.add(task)
        task.add_done_callback(self._fact_check_tasks.discard)
        return

    async def _run_fact_check_job(
        self,
        event: AstrMessageEvent,
        request_data: FactCheckRequest,
        started_at: float,
        cache_key: str,
        *,
        pipeline_task: asyncio.Task[FactCheckResult] | None = None,
        joined_existing: bool = False,
    ) -> None:
        label = self._event_label(event)
        timeout_seconds = max(
            10.0,
            float(self.config.get("fact_check_total_timeout_seconds") or 90),
        )

        try:
            if pipeline_task is None:
                async def compute() -> FactCheckResult:
                    return await run_blocking_with_timeout(
                        partial(
                            self._run_fact_check_sync,
                            request_data,
                            timeout_seconds,
                        ),
                        timeout=timeout_seconds,
                        capacity=self._fact_check_semaphore,
                    )

                result, joined_existing = await self._singleflight.run(
                    cache_key, compute
                )
            else:
                result = await asyncio.shield(pipeline_task)
            if joined_existing:
                logger.info(
                    f"[astrbot-fact-check-singleflight-hit] {label}: key={cache_key[:12]}"
                )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - started_at
            reason = f"timeout after {elapsed:.1f}s"
            self._record_fact_check_metric(
                success=False,
                elapsed=elapsed,
                failure_stage="timeout",
            )
            logger.error(f"[astrbot-fact-check-error] {label}: {reason}")
            logger.info(f"[astrbot-fact-check-reason] {label}: {reason}")
            await self._send_fact_check_reply(
                event,
                self._failed_fact_check_reply(reason),
                label=label,
                purpose="timeout",
            )
            return
        except Exception as exc:
            reason = f"exception: {exc!r}"
            self._record_fact_check_metric(
                success=False,
                elapsed=time.perf_counter() - started_at,
                failure_stage="exception",
            )
            logger.error(f"[astrbot-fact-check-error] {label}: {exc!r}")
            logger.info(f"[astrbot-fact-check-reason] {label}: {reason}")
            self._maybe_start_cooldown(reason)
            await self._send_fact_check_reply(
                event,
                self._failed_fact_check_reply(reason),
                label=label,
                purpose="exception",
            )
            return

        if result.reason and not result.reason.startswith("ok"):
            logger.info(f"[astrbot-fact-check-reason] {label}: {result.reason}")
        logger.info(
            f"[astrbot-fact-check-done] {label}: "
            f"{time.perf_counter() - started_at:.2f}s",
        )
        successful = self._is_successful_result(result)
        self._record_fact_check_metric(
            success=successful,
            elapsed=time.perf_counter() - started_at,
            partial=str(result.reason or "").startswith("ok; partial"),
            failure_stage="pipeline" if not successful else "",
        )
        session_id = None
        if successful:
            self._set_cached_result(cache_key, result, request_data=request_data)
            session_id = self._remember_fact_check_session(event, request_data, result)
        await self._send_fact_check_reply(
            event,
            result.reply or FAILED_REPLY,
            label=label,
            purpose="result",
            session_id=session_id,
        )

    def _run_fact_check_sync(
        self,
        request_data: FactCheckRequest,
        timeout_seconds: float,
    ) -> FactCheckResult:
        return run_fact_check(
            **build_fact_check_kwargs(
                self.config,
                request_data,
                timeout_seconds,
                list_config=self._list_config,
            ),
        )

    async def _send_fact_check_reply(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        label: str,
        purpose: str,
        session_id: str | None = None,
    ) -> bool:
        sent = await self._send_fact_check_reply_impl(
            event,
            text,
            label=label,
            purpose=purpose,
            session_id=session_id,
        )
        store = getattr(self, "_metrics_store", None)
        if isinstance(store, FactCheckMetricsStore):
            store.record_delivery(success=sent)
        return sent

    async def _send_fact_check_reply_impl(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        label: str,
        purpose: str,
        session_id: str | None = None,
    ) -> bool:
        text = str(text or FAILED_REPLY).strip() or FAILED_REPLY
        logger.info(
            f"[astrbot-fact-check-send] {label}: purpose={purpose} len={len(text)}",
        )
        forward_result = self._fact_check_forward_result(
            event, text, session_id=session_id
        )
        safe_text = self._sanitize_forward_text_for_qq(text)
        retry_result = (
            self._fact_check_forward_result(event, safe_text, session_id=session_id)
            if safe_text != text
            else forward_result
        )

        outcome = (
            await send_chain_result(event, forward_result)
            if send_chain_result
            else None
        )
        try:
            if outcome is None:
                await event.send(forward_result)
            elif outcome.assumed_ok:
                logger.info(
                    f"[astrbot-fact-check-send-assume-ok] {label}: "
                    f"method=event.send.forward kind={outcome.kind} error={outcome.error}",
                )
                return True
            elif not outcome.ok:
                raise RuntimeError(f"{outcome.kind}: {outcome.error}")
            logger.info(
                f"[astrbot-fact-check-send-ok] {label}: method=event.send.forward"
            )
            return True
        except Exception as exc:
            if self._looks_like_confirm_timeout(exc):
                logger.info(
                    f"[astrbot-fact-check-send-assume-ok] {label}: "
                    f"method=event.send.forward error={exc!r}",
                )
                return True
            logger.warning(
                f"[astrbot-fact-check-send-error] {label}: "
                f"method=event.send.forward kind={getattr(outcome, 'kind', None) or 'unknown'} error={exc!r}",
            )

        if safe_text != text:
            await asyncio.sleep(1.0)
            outcome = (
                await send_chain_result(event, retry_result)
                if send_chain_result
                else None
            )
            try:
                logger.info(
                    f"[astrbot-fact-check-send-retry-sanitized] {label}: "
                    f"len={len(text)}->{len(safe_text)}",
                )
                if outcome is None:
                    await event.send(retry_result)
                elif outcome.assumed_ok:
                    logger.info(
                        f"[astrbot-fact-check-send-assume-ok] {label}: "
                        f"method=event.send.forward retry kind={outcome.kind} error={outcome.error}",
                    )
                    return True
                elif not outcome.ok:
                    raise RuntimeError(f"{outcome.kind}: {outcome.error}")
                logger.info(
                    f"[astrbot-fact-check-send-ok] {label}: method=event.send.forward retry"
                )
                return True
            except Exception as exc:
                if self._looks_like_confirm_timeout(exc):
                    logger.info(
                        f"[astrbot-fact-check-send-assume-ok] {label}: "
                        f"method=event.send.forward retry error={exc!r}",
                    )
                    return True
                logger.error(
                    f"[astrbot-fact-check-send-failed] {label}: "
                    f"method=event.send.forward retry kind={getattr(outcome, 'kind', None) or 'unknown'} error={exc!r}",
                )
                self._dump_forward_failure(label, text, exc)
        else:
            logger.info(
                f"[astrbot-fact-check-send-skip-retry] {label}: "
                "sanitized forward is identical",
            )

        fallback_text = self._fact_check_text_with_session_marker(
            safe_text, session_id=session_id
        )
        if await self._send_text_via_onebot(
            event,
            fallback_text,
            label=label,
            prefer_send_msg=True,
            suppress_errors=True,
        ):
            logger.info(f"[astrbot-fact-check-send-ok] {label}: method=onebot fallback")
            return True
        try:
            await event.send(event.plain_result(fallback_text))
            logger.info(f"[astrbot-fact-check-send-ok] {label}: method=plain fallback")
            return True
        except Exception as fallback_exc:
            logger.error(
                f"[astrbot-fact-check-send-failed] {label}: "
                f"method=plain fallback error={fallback_exc!r}",
            )
            return False

    async def _send_text_via_onebot(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        label: str,
        force_private: bool = False,
        suppress_errors: bool = False,
        prefer_send_msg: bool = False,
    ) -> bool:
        bot = getattr(event, "bot", None)
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            logger.warning(
                f"[astrbot-fact-check-send-skip] {label}: no onebot call_action"
            )
            return False

        group_id = "" if force_private else str(event.get_group_id() or "").strip()
        user_id = str(event.get_sender_id() or "").strip()
        if prefer_send_msg and group_id:
            action = "send_msg"
            target_key = "group_id"
        else:
            action = "send_group_msg" if group_id else "send_private_msg"
            target_key = "group_id" if group_id else "user_id"
        target_value = group_id or user_id
        if not target_value.isdigit():
            logger.warning(
                f"[astrbot-fact-check-send-skip] {label}: invalid target={target_value!r}"
            )
            return False

        chunks = self._split_reply_text(text, max_chars=350 if group_id else 700)
        next_index = 0
        for attempt in range(1, 4):
            try:
                for index in range(next_index, len(chunks)):
                    chunk = chunks[index]
                    payload = {
                        target_key: int(target_value),
                        "message": [{"type": "text", "data": {"text": chunk}}],
                    }
                    if action == "send_msg":
                        payload["message_type"] = "group"
                    await call_action(action, **payload)
                    next_index = index + 1
                    if next_index < len(chunks):
                        await asyncio.sleep(1.0)
                logger.info(
                    f"[astrbot-fact-check-send-ok] {label}: "
                    f"method=onebot action={action} chunks={len(chunks)} attempt={attempt}",
                )
                return True
            except Exception as exc:
                if self._looks_like_confirm_timeout(exc):
                    next_index = min(next_index + 1, len(chunks))
                    logger.info(
                        f"[astrbot-fact-check-send-assume-ok] {label}: "
                        f"method=onebot action={action} chunks={len(chunks)} "
                        f"sent={next_index} attempt={attempt} error={exc!r}",
                    )
                    if next_index >= len(chunks):
                        return True
                    continue
                message = (
                    f"[astrbot-fact-check-send-error] {label}: "
                    f"method=onebot action={action} chunks={len(chunks)} "
                    f"sent={next_index} attempt={attempt}/3 error={exc!r}"
                )
                if suppress_errors:
                    logger.info(message)
                else:
                    logger.warning(message)
                await asyncio.sleep(1.0 * attempt)
        return False

    def _looks_like_confirm_timeout(self, exc: Exception) -> bool:
        if is_confirm_timeout is not None:
            return bool(is_confirm_timeout(exc))
        text = str(exc)
        return (
            "Timeout: NTEvent" in text
            and '"result": 0' in text
            and '"errMsg": ""' in text
        )

    def _sanitize_forward_text_for_qq(self, text: str) -> str:
        # A fact-check must not change names or claims merely to satisfy one QQ
        # transport. The caller already falls back to segmented plain text.
        return str(text or "")

    def _fact_check_forward_result(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        session_id: str | None = None,
    ):
        chunks = self._split_reply_text(text, max_chars=1200)
        if session_id:
            marker = f"\n\n追问可回复本消息。核查ID：{session_id}"
            if chunks:
                chunks[-1] = (chunks[-1].rstrip() + marker).strip()
        if len(chunks) == 1:
            chunk = chunks[0]
            split_at = self._fact_check_single_node_split_at(chunk)
            chunks = [chunk[:split_at].strip(), chunk[split_at:].strip()]
            chunks = [item for item in chunks if item]
        nodes = []
        self_id = str(event.get_self_id() or "0")
        for index, chunk in enumerate(chunks, start=1):
            name = "事实核查" if len(chunks) == 1 else f"事实核查 {index}/{len(chunks)}"
            nodes.append(Node(uin=self_id, name=name, content=[Plain(chunk)]))
        logger.info(
            f"[astrbot-fact-check-forward-build] nodes={len(nodes)} "
            f"lengths={[len(chunk) for chunk in chunks]}",
        )
        return event.chain_result([Nodes(nodes)])

    @staticmethod
    def _fact_check_text_with_session_marker(
        text: str, *, session_id: str | None
    ) -> str:
        text = str(text or FAILED_REPLY).strip() or FAILED_REPLY
        if not session_id:
            return text
        if session_id in text:
            return text
        return (text.rstrip() + f"\n\n追问可回复本消息。核查ID：{session_id}").strip()

    def _dump_forward_failure(self, label: str, text: str, exc: Exception) -> None:
        try:
            chunks = self._split_reply_text(text, max_chars=1200)
            path = Path(StarTools.get_data_dir()) / "last_forward_failure.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "label": label,
                "length": len(text),
                "chunk_lengths": [len(chunk) for chunk in chunks],
                "error": type(exc).__name__,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            if bool(self.config.get("fact_check_debug_store_full_failure_text", False)):
                payload["text"] = text
                payload["chunks"] = chunks
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"[astrbot-fact-check-forward-failure-dump] {label}: {path}")
        except Exception as dump_exc:
            logger.warning(
                f"[astrbot-fact-check-forward-failure-dump-error] {label}: {dump_exc!r}"
            )

    def _cleanup_forward_failure_dump(self) -> None:
        path = Path(StarTools.get_data_dir()) / "last_forward_failure.json"
        try:
            if path.is_file() and time.time() - path.stat().st_mtime > 24 * 60 * 60:
                path.unlink()
        except OSError as exc:
            logger.warning(
                f"[astrbot-fact-check-forward-failure-cleanup-error] {exc!r}"
            )

    def _request_cache_key(self, request_data: FactCheckRequest) -> str:
        def cache_config_value(key: str, default):
            value = self.config.get(key, None)
            if value is None or value == "":
                return default
            return value

        image_records: list[dict[str, str]] = []
        for image in request_data.images:
            digest = image.content_sha256 or self._image_input_digest(image)
            image_records.append(
                {
                    "content_sha256": digest,
                    "url": "" if digest else image.url,
                    "file_name": "" if digest else image.file_name,
                },
            )
        anysearch_api_key = str(
            self.config.get("fact_check_anysearch_api_key") or ""
        ).strip()
        payload = {
            "pipeline_version": FACT_CHECK_PIPELINE_VERSION,
            "text": request_data.text.strip(),
            "speaker": request_data.speaker.strip(),
            "images": image_records,
            "models": {
                "pre": str(
                    self.config.get("fact_check_pre_model") or "gemini-3.1-flash-lite"
                ).strip(),
                "evidence": str(
                    self.config.get("fact_check_evidence_model") or "gemini-2.5-flash"
                ).strip(),
                "evidence_max_output_tokens": str(
                    cache_config_value("fact_check_evidence_max_output_tokens", 1536),
                ),
                "evidence_retry_max_output_tokens": str(
                    cache_config_value(
                        "fact_check_evidence_retry_max_output_tokens", 3072
                    ),
                ),
                "verdict": self._list_config(
                    "fact_check_verdict_models",
                    ["gemini-3-flash-preview"],
                ),
                "verdict_timeout_seconds": str(
                    cache_config_value("fact_check_verdict_timeout_seconds", 25),
                ),
                "verdict_max_output_tokens": str(
                    cache_config_value("fact_check_verdict_max_output_tokens", 2048),
                ),
                "verdict_retry_max_output_tokens": str(
                    cache_config_value(
                        "fact_check_verdict_retry_max_output_tokens", 4096
                    ),
                ),
                "verdict_policy": str(
                    cache_config_value("fact_check_verdict_policy", "risk_based"),
                ),
                "verdict_thinking_level": str(
                    cache_config_value("fact_check_verdict_thinking_level", "medium"),
                ),
            },
            "image_config": {
                "max_image_bytes": str(
                    cache_config_value("fact_check_max_image_bytes", 5 * 1024 * 1024),
                ),
                "download_hard_limit_bytes": str(
                    cache_config_value(
                        "fact_check_image_download_hard_limit_bytes",
                        20 * 1024 * 1024,
                    ),
                ),
                "max_pixels": str(
                    cache_config_value("fact_check_image_max_pixels", 20_000_000),
                ),
                "total_inline_bytes": str(
                    cache_config_value(
                        "fact_check_image_total_inline_bytes",
                        10 * 1024 * 1024,
                    ),
                ),
            },
            "anysearch": {
                "enabled": bool(self.config.get("fact_check_anysearch_enabled", False)),
                "endpoint": str(
                    self.config.get("fact_check_anysearch_endpoint")
                    or "https://api.anysearch.com/mcp",
                ).strip(),
                "api_key_sha256": hashlib.sha256(
                    anysearch_api_key.encode("utf-8")
                ).hexdigest()
                if anysearch_api_key
                else "",
                "timeout_seconds": str(
                    cache_config_value("fact_check_anysearch_timeout_seconds", 20)
                ),
                "max_claims": str(
                    cache_config_value("fact_check_anysearch_max_claims", 3)
                ),
                "max_results_per_claim": str(
                    cache_config_value("fact_check_anysearch_max_results_per_claim", 3),
                ),
                "extract_top_urls": str(
                    cache_config_value("fact_check_anysearch_extract_top_urls", 2)
                ),
                "max_chars": str(
                    cache_config_value("fact_check_anysearch_max_chars", 6000)
                ),
                "freshness": str(
                    self.config.get("fact_check_anysearch_freshness") or ""
                ).strip(),
                "content_types": self._list_config(
                    "fact_check_anysearch_content_types",
                    ["web", "news"],
                ),
            },
        }
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_successful_result(result: FactCheckResult) -> bool:
        return not result.reason or result.reason.startswith("ok")

    @staticmethod
    def _merge_sources(
        preferred: Iterable[str],
        existing: Iterable[str],
        *,
        limit: int,
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for source in [*preferred, *existing]:
            item = str(source or "").strip()
            key = re.sub(r"\s+", "", item).lower()
            if not item or key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max(1, int(limit or 1)):
                break
        return merged

    @staticmethod
    def _image_input_digest(image: ImageInput) -> str:
        if image.content_sha256:
            return image.content_sha256
        path = Path(
            str(image.path or "").removeprefix("file:///").removeprefix("file://")
        )
        if not path.is_file():
            return ""
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return ""

    def _dedupe_image_inputs(self, images: Iterable[ImageInput]) -> list[ImageInput]:
        unique: list[ImageInput] = []
        seen: set[str] = set()
        for image in images:
            digest = image.content_sha256 or self._image_input_digest(image)
            key = (
                f"sha256:{digest}" if digest else f"url:{str(image.url or '').strip()}"
            )
            if key in seen:
                continue
            seen.add(key)
            if digest and not image.content_sha256:
                image.content_sha256 = digest
            unique.append(image)
        return unique

    def _is_fact_check_allowed(self, event: AstrMessageEvent) -> bool:
        if is_plugin_allowed is None:
            return bool(self.config.get("fact_check_access_control_fail_open", False))
        return bool(
            is_plugin_allowed(
                "fact_check",
                event,
                default_allow=False,
                default_allow_private=bool(
                    self.config.get("fact_check_default_allow_private", True),
                ),
            )
        )

    def _max_fact_check_queue(self) -> int:
        return max(1, int(self.config.get("fact_check_max_queue") or 4))

    def _active_fact_check_jobs(self) -> int:
        waiting_events = len(getattr(self, "_fact_check_tasks", set()) or set())
        pipeline_jobs = int(
            getattr(getattr(self, "_singleflight", None), "active_count", 0) or 0
        )
        return max(waiting_events, pipeline_jobs) + max(
            0,
            int(getattr(self, "_active_followup_jobs", 0)),
        )

    def _fact_check_queue_full(self) -> bool:
        return self._active_fact_check_jobs() >= self._max_fact_check_queue()

    def _record_fact_check_metric(
        self,
        *,
        success: bool,
        elapsed: float,
        cache_hit: bool = False,
        followup: bool = False,
        partial: bool = False,
        failure_stage: str = "",
    ) -> None:
        store = getattr(self, "_metrics_store", None)
        if not isinstance(store, FactCheckMetricsStore):
            store = FactCheckMetricsStore(None)
            self._metrics_store = store
        outcome = "partial" if partial else ("success" if success else "failure")
        metrics = store.record(
            outcome=outcome,
            elapsed=elapsed,
            cache_hit=cache_hit,
            followup=followup,
            failure_stage=failure_stage,
        )
        requests = int(metrics["requests"])
        log_every = max(
            1,
            int(self.config.get("fact_check_metrics_log_every") or 20),
        )
        if requests % log_every:
            return
        logger.info(
            "[astrbot-fact-check-metrics] "
            f"requests={requests} "
            f"pipeline_success={int(metrics.get('success', 0))} "
            f"pipeline_partial={int(metrics.get('partial', 0))} "
            f"pipeline_failure={int(metrics.get('failure', 0))} "
            f"cache_hits={int(metrics.get('cache_hits', 0))} "
            f"followups={int(metrics.get('followups', 0))} "
            f"avg_seconds={float(metrics.get('average_seconds', 0.0)):.2f}",
        )

    def _remember_fact_check_session(
        self,
        event: AstrMessageEvent,
        request_data: FactCheckRequest,
        result: FactCheckResult,
    ) -> str:
        self._cleanup_fact_check_sessions()
        session_id = "fc_" + uuid.uuid4().hex[:8]
        now = time.time()
        self._fact_check_sessions[session_id] = FactCheckSession(
            session_id=session_id,
            created_at=now,
            group_id=str(event.get_group_id() or "").strip(),
            user_id=str(event.get_sender_id() or "").strip(),
            request_data=request_data,
            reply=result.reply or FAILED_REPLY,
            updated_at=now,
            candidates=list(result.candidates or []),
            sources=list(result.sources or []),
        )
        self._cleanup_fact_check_sessions()
        self._persist_fact_check_sessions()
        logger.info(
            f"[astrbot-fact-check-session-save] {self._event_label(event)}: session={session_id}"
        )
        return session_id

    def _session_store_path(self) -> Path:
        return Path(StarTools.get_data_dir()) / "fact_check_sessions.json"

    def _persist_fact_check_sessions(self) -> None:
        if not bool(getattr(self, "_session_store_enabled", False)):
            return
        path = self._session_store_path()
        payload = {
            "version": 2,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at or session.created_at,
                    "group_id": session.group_id,
                    "user_id": session.user_id,
                    "reply": session.reply,
                    "candidates": [
                        {
                            "claim": str(getattr(candidate, "claim", "") or ""),
                            "source": str(getattr(candidate, "source", "") or ""),
                            "priority": int(getattr(candidate, "priority", 3) or 3),
                        }
                        for candidate in session.candidates
                        if str(getattr(candidate, "claim", "") or "").strip()
                    ],
                    "sources": list(session.sources),
                }
                for session in sorted(
                    self._fact_check_sessions.values(),
                    key=lambda item: item.created_at,
                )
            ],
        }
        try:
            atomic_write_json(path, payload)
        except OSError as exc:
            logger.warning(f"[astrbot-fact-check-session-persist-error] {exc!r}")

    def _load_fact_check_sessions(self) -> None:
        if not bool(getattr(self, "_session_store_enabled", False)):
            return
        path = self._session_store_path()
        if not path.is_file():
            return
        try:
            payload = read_json_file(path, {})
            records = payload.get("sessions") if isinstance(payload, dict) else []
            loaded: dict[str, FactCheckSession] = {}
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                session_id = str(record.get("session_id") or "").strip()
                if not re.fullmatch(r"fc_[0-9a-fA-F]{8,16}", session_id):
                    continue
                candidates = [
                    ClaimCandidate(
                        claim=str(item.get("claim") or "").strip(),
                        source=str(item.get("source") or "").strip(),
                        priority=max(1, min(5, int(item.get("priority") or 3))),
                    )
                    for item in record.get("candidates") or []
                    if isinstance(item, dict) and str(item.get("claim") or "").strip()
                ]
                loaded[session_id] = FactCheckSession(
                    session_id=session_id,
                    created_at=float(record.get("created_at") or 0),
                    group_id=str(record.get("group_id") or "").strip(),
                    user_id=str(record.get("user_id") or "").strip(),
                    request_data=FactCheckRequest(text="", trigger_text=""),
                    reply=str(record.get("reply") or FAILED_REPLY),
                    updated_at=float(
                        record.get("updated_at") or record.get("created_at") or 0
                    ),
                    candidates=candidates,
                    sources=[
                        str(source).strip()
                        for source in record.get("sources") or []
                        if str(source).strip()
                    ],
                )
            self._fact_check_sessions = loaded
            if self._cleanup_fact_check_sessions():
                self._persist_fact_check_sessions()
            logger.info(
                f"[astrbot-fact-check-session-load] sessions={len(self._fact_check_sessions)}",
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(f"[astrbot-fact-check-session-load-error] {exc!r}")

    def _cleanup_fact_check_sessions(self) -> bool:
        ttl = max(60, int(self.config.get("fact_check_followup_ttl_seconds") or 3600))
        max_entries = max(
            8, int(self.config.get("fact_check_followup_max_sessions") or 50)
        )
        now = time.time()
        expired = [
            session_id
            for session_id, session in self._fact_check_sessions.items()
            if now - (session.updated_at or session.created_at) > ttl
        ]
        for session_id in expired:
            self._fact_check_sessions.pop(session_id, None)
        changed = bool(expired)
        if len(self._fact_check_sessions) <= max_entries:
            return changed
        stale = sorted(
            self._fact_check_sessions.values(),
            key=lambda item: item.updated_at or item.created_at,
        )
        for session in stale[: len(self._fact_check_sessions) - max_entries]:
            self._fact_check_sessions.pop(session.session_id, None)
            changed = True
        return changed

    def _extract_followup_question(self, event: AstrMessageEvent) -> str:
        has_reply = any(isinstance(comp, Reply) for comp in event.get_messages())
        if not has_reply:
            return ""
        for text in _event_text_candidates(event):
            cleaned = re.sub(r"fc_[0-9a-fA-F]{8,16}", " ", text)
            cleaned = re.sub(r"核查ID[:：]\s*", " ", cleaned)
            cleaned = cleaned.strip(" \t\r\n:：")
            if cleaned and not self._is_unusable_quoted_text(cleaned):
                return cleaned[:800]
        return ""

    async def _find_followup_session(
        self, event: AstrMessageEvent
    ) -> FactCheckSession | None:
        session, _ = await self._find_followup_session_with_state(event)
        return session

    async def _find_followup_session_with_state(
        self,
        event: AstrMessageEvent,
    ) -> tuple[FactCheckSession | None, bool]:
        if self._cleanup_fact_check_sessions():
            self._persist_fact_check_sessions()
        ids: list[str] = []
        quoted_looks_like_fact_check = False
        quoted_fact_check_texts: list[str] = []
        replies_without_local_text: list[Reply] = []
        for text in _event_text_candidates(event):
            ids.extend(self._extract_fact_check_session_ids(text))

        for comp in event.get_messages():
            if not isinstance(comp, Reply):
                continue
            local_texts = [str(comp.message_str or "")] + self._plain_texts(
                comp.chain or []
            )
            usable_local_texts = [
                text
                for text in local_texts
                if text.strip() and not self._is_unusable_quoted_text(text)
            ]
            if not usable_local_texts:
                replies_without_local_text.append(comp)
            for text in usable_local_texts:
                ids.extend(self._extract_fact_check_session_ids(text))
                if self._looks_like_fact_check_reply(text):
                    quoted_looks_like_fact_check = True
                    quoted_fact_check_texts.append(text)

        for session_id in ids:
            session = self._fact_check_sessions.get(session_id)
            if session:
                return session, False
        if ids:
            return None, True

        candidates = [
            session
            for session in self._fact_check_sessions.values()
            if self._session_visible_to_event(session, event)
        ]
        if not candidates:
            return None, quoted_looks_like_fact_check

        def match_quoted_text() -> tuple[FactCheckSession | None, bool]:
            if not quoted_fact_check_texts:
                return None, False
            ranked = sorted(
                (
                    (
                        max(
                            (
                                self._fact_check_reply_match_score(text, session.reply)
                                for text in quoted_fact_check_texts
                            ),
                            default=0.0,
                        ),
                        session,
                    )
                    for session in candidates
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            best_score, best_session = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score < 0.62 or (
                len(ranked) > 1 and best_score - second_score < 0.08
            ):
                return None, True
            return best_session, False

        # NapCat usually includes enough quoted text locally. Match that before
        # asking OneBot for the full message to avoid a network call on every reply.
        if quoted_looks_like_fact_check:
            return match_quoted_text()

        if not replies_without_local_text:
            return None, False

        for comp in replies_without_local_text:
            fetched = await self._fetch_reply_payload(event, comp)
            if not fetched:
                continue
            fetched_texts, _, _ = fetched
            for text in fetched_texts:
                ids.extend(self._extract_fact_check_session_ids(text))
                if self._looks_like_fact_check_reply(text):
                    quoted_looks_like_fact_check = True
                    quoted_fact_check_texts.append(text)

        for session_id in ids:
            session = self._fact_check_sessions.get(session_id)
            if session:
                return session, False
        if ids:
            return None, True
        if not quoted_looks_like_fact_check:
            return None, False

        return match_quoted_text()

    @staticmethod
    def _fact_check_reply_match_score(quoted: str, saved: str) -> float:
        def normalize(value: str) -> str:
            value = re.sub(r"fc_[0-9a-fA-F]{8,16}", "", str(value or ""))
            value = re.sub(r"核查ID[:：]?", "", value)
            value = re.sub(r"追问可回复本消息[。.]?", "", value)
            return re.sub(r"[^\w\u4e00-\u9fff]+", "", value).lower()[:4000]

        left = normalize(quoted)
        right = normalize(saved)
        if not left or not right:
            return 0.0
        if len(left) >= 40 and left in right:
            return min(1.0, 0.85 + len(left) / max(len(right), 1) * 0.15)
        if len(right) >= 40 and right in left:
            return min(1.0, 0.85 + len(right) / max(len(left), 1) * 0.15)
        return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()

    @staticmethod
    def _extract_fact_check_session_ids(text: str | None) -> list[str]:
        if not text:
            return []
        return [
            match.group(0).lower()
            for match in re.finditer(r"fc_[0-9a-fA-F]{8,16}", str(text))
        ]

    @staticmethod
    def _looks_like_fact_check_reply(text: str | None) -> bool:
        normalized = str(text or "")
        if not normalized.strip():
            return False
        if FactCheckPlugin._is_fact_check_command_only(normalized):
            return False
        if "核查ID" in normalized or "追问可回复本消息" in normalized:
            return True
        if re.search(r"事实核查\s*(?:\d+\s*/\s*\d+)?\s*[:：]", normalized):
            return True
        if "事实核查" in normalized and any(
            marker in normalized for marker in ("要点：", "来源：", "总结论")
        ):
            return True
        return False

    @staticmethod
    def _session_visible_to_event(
        session: FactCheckSession, event: AstrMessageEvent
    ) -> bool:
        group_id = str(event.get_group_id() or "").strip()
        sender_id = str(event.get_sender_id() or "").strip()
        if session.group_id:
            return bool(
                group_id
                and group_id == session.group_id
                and sender_id
                and sender_id == session.user_id
            )
        return bool(sender_id and sender_id == session.user_id)

    def _get_cached_result(self, cache_key: str) -> FactCheckResult | None:
        cached = self._reply_cache.get(cache_key)
        if not cached:
            return None
        if len(cached) == 3:
            created_at, ttl, value = cached
        else:
            created_at, value = cached  # type: ignore[misc]
            ttl = max(0, int(self.config.get("fact_check_cache_ttl_seconds") or 600))
        if ttl <= 0:
            self._reply_cache.pop(cache_key, None)
            return None
        if time.time() - created_at > ttl:
            self._reply_cache.pop(cache_key, None)
            return None
        if isinstance(value, FactCheckResult):
            return value
        return FactCheckResult(str(value or FAILED_REPLY), "ok; cache", [], [])

    def _cache_ttl_for_request(
        self, request_data: FactCheckRequest | None = None
    ) -> int:
        ttl = max(0, int(self.config.get("fact_check_cache_ttl_seconds") or 600))
        if request_data is None:
            return ttl
        freshness = infer_anysearch_freshness(request_data.text)
        if freshness == "day":
            return min(ttl, 120)
        if freshness == "week":
            return min(ttl, 300)
        return ttl

    def _set_cached_result(
        self,
        cache_key: str,
        result: FactCheckResult,
        *,
        request_data: FactCheckRequest | None = None,
    ) -> None:
        ttl = self._cache_ttl_for_request(request_data)
        if ttl <= 0:
            return
        self._reply_cache[cache_key] = (
            time.time(),
            float(ttl),
            FactCheckResult(
                reply=result.reply or FAILED_REPLY,
                reason=result.reason or "ok; cache",
                sources=list(result.sources or []),
                candidates=list(result.candidates or []),
            ),
        )
        max_entries = max(8, int(self.config.get("fact_check_cache_max_entries") or 32))
        while len(self._reply_cache) > max_entries:
            oldest_key = ""
            oldest_at = float("inf")
            for key, cached in self._reply_cache.items():
                created_at = cached[0]
                if created_at < oldest_at:
                    oldest_key = key
                    oldest_at = created_at
            if not oldest_key:
                break
            self._reply_cache.pop(oldest_key, None)

    def _get_cached_reply(self, cache_key: str) -> str:
        result = self._get_cached_result(cache_key)
        return result.reply if result else ""

    def _set_cached_reply(self, cache_key: str, reply: str) -> None:
        self._set_cached_result(
            cache_key, FactCheckResult(str(reply or FAILED_REPLY), "ok; cache", [], [])
        )

    def _cooldown_left(self) -> float:
        return max(0.0, self._cooldown_until - time.time())

    def _maybe_start_cooldown(self, reason: str) -> None:
        lowered = str(reason or "").lower()
        capacity_markers = (
            "429",
            "503",
            "too many requests",
            "rate limit",
            "cooling down",
            "cooldown",
            "high demand",
            "resource_exhausted",
            "temporarily unavailable",
        )
        if not any(marker in lowered for marker in capacity_markers):
            return
        seconds = max(
            0, int(self.config.get("fact_check_rate_limit_cooldown_seconds") or 90)
        )
        if seconds <= 0:
            return
        self._cooldown_until = max(self._cooldown_until, time.time() + seconds)
        logger.warning(f"[astrbot-fact-check-rate-limit-cooldown] seconds={seconds}")

    def _fact_check_single_node_split_at(self, text: str) -> int:
        text = str(text or "")
        midpoint = max(1, len(text) // 2)
        candidates = [
            text.find("\n来源", 1),
            text.find("\n证据", 1),
            text.find("\n要点", 1),
            text.find("\n结论", 1),
        ]
        candidates = [pos for pos in candidates if pos > 0]
        if candidates:
            return min(candidates, key=lambda pos: abs(pos - midpoint))
        newline_before = text.rfind("\n", 0, midpoint)
        newline_after = text.find("\n", midpoint)
        newline_candidates = [pos for pos in (newline_before, newline_after) if pos > 0]
        if newline_candidates:
            return min(newline_candidates, key=lambda pos: abs(pos - midpoint))
        return midpoint

    def _split_reply_text(self, text: str, *, max_chars: int) -> list[str]:
        text = str(text or "").strip()
        if not text:
            return [FAILED_REPLY]
        chunks: list[str] = []
        current = ""
        for line in text.splitlines():
            addition = line if not current else "\n" + line
            if len(current) + len(addition) <= max_chars:
                current += addition
                continue
            if current:
                chunks.append(current)
            while len(line) > max_chars:
                chunks.append(line[:max_chars])
                line = line[max_chars:]
            current = line
        if current:
            chunks.append(current)
        return chunks or [FAILED_REPLY]

    @filter.command("factcheck")
    async def factcheck_help(self, event: AstrMessageEvent):
        """Show fact-check usage when the bare command is used."""
        event.set_extra("qq_agent_command_handled", True)
        event.stop_event()
        yield event.plain_result(self._fact_check_usage_text())

    @filter.command("事实核查状态")
    async def factcheck_status(self, event: AstrMessageEvent):
        """Show persisted aggregate quality and performance counters."""
        event.set_extra("qq_agent_command_handled", True)
        event.stop_event()
        if not self._is_fact_check_allowed(event):
            yield event.plain_result("这个群没开事实核查。")
            return
        store = getattr(self, "_metrics_store", None)
        if not isinstance(store, FactCheckMetricsStore):
            store = FactCheckMetricsStore(None)
            self._metrics_store = store
        yield event.plain_result(store.render_status())

    @staticmethod
    def _fact_check_usage_text() -> str:
        return (
            "用法：回复一条消息后发送 /事实核查，或者直接发送 /事实核查 要核查的内容。"
        )

    async def _build_fact_check_request(
        self,
        event: AstrMessageEvent,
        *,
        trigger_text: str | None = None,
    ) -> FactCheckRequest:
        trigger_text = trigger_text or _trigger_text(event) or event.message_str or ""
        inline_text = remove_trigger(trigger_text)
        if re.fullmatch(r"(?:@\S+\s*)+", inline_text or ""):
            inline_text = ""
        speaker = ""
        quoted_texts: list[str] = []
        images: list[ImageInput] = []
        seen_image_refs: set[str] = set()
        max_images = max(0, int(self.config.get("fact_check_max_images") or 2))
        collect_limit = max(max_images + 4, max_images * 3)

        for comp in event.get_messages():
            if isinstance(comp, Reply):
                speaker = str(comp.sender_nickname or comp.sender_id or "").strip()
                before_text_count = len(quoted_texts)
                before_image_count = len(images)
                local_forward_ids: list[str] = []
                local_texts: list[str] = []
                if comp.message_str:
                    comp_text = str(comp.message_str).strip()
                    local_texts.append(comp_text)
                    local_forward_ids.extend(
                        self._extract_forward_ids_from_text(comp_text)
                    )
                if comp.chain:
                    local_texts.extend(self._plain_texts(comp.chain))
                    local_forward_ids.extend(
                        self._extract_forward_ids_from_components(comp.chain)
                    )
                    images.extend(
                        await self._image_inputs(
                            comp.chain,
                            remaining=collect_limit - len(images),
                            seen_refs=seen_image_refs,
                        ),
                    )
                local_text = "\n".join(part for part in local_texts if part).strip()
                if local_text and not self._is_unusable_quoted_text(local_text):
                    quoted_texts.append(local_text)

                if (
                    len(quoted_texts) == before_text_count
                    or self._is_unusable_quoted_text(local_text)
                    or len(images) == before_image_count
                ):
                    fetched = await self._fetch_reply_payload(event, comp)
                    if fetched:
                        fetched_texts, fetched_images, fetched_speaker = fetched
                        if fetched_speaker and not speaker:
                            speaker = fetched_speaker
                        for fetched_text in fetched_texts:
                            if fetched_text and fetched_text not in quoted_texts:
                                quoted_texts.append(fetched_text)
                        images.extend(
                            await self._image_inputs(
                                fetched_images,
                                remaining=collect_limit - len(images),
                                seen_refs=seen_image_refs,
                            ),
                        )
                if local_forward_ids:
                    speaker = await self._append_forward_payloads(
                        event,
                        local_forward_ids,
                        quoted_texts=quoted_texts,
                        images=images,
                        max_images=collect_limit,
                        speaker=speaker,
                        label=f"reply:{getattr(comp, 'id', '')}",
                        seen_image_refs=seen_image_refs,
                    )
            elif isinstance(comp, Forward):
                speaker = await self._append_forward_payloads(
                    event,
                    [str(getattr(comp, "id", "") or "")],
                    quoted_texts=quoted_texts,
                    images=images,
                    max_images=collect_limit,
                    speaker=speaker,
                    label="direct",
                    seen_image_refs=seen_image_refs,
                )
            elif isinstance(comp, Plain):
                forward_ids = self._extract_forward_ids_from_text(str(comp.text or ""))
                if forward_ids:
                    speaker = await self._append_forward_payloads(
                        event,
                        forward_ids,
                        quoted_texts=quoted_texts,
                        images=images,
                        max_images=collect_limit,
                        speaker=speaker,
                        label="plain",
                        seen_image_refs=seen_image_refs,
                    )
            elif isinstance(comp, Image):
                images.extend(
                    await self._image_inputs(
                        [comp],
                        remaining=collect_limit - len(images),
                        seen_refs=seen_image_refs,
                    ),
                )

            if len(images) >= collect_limit:
                images = images[:collect_limit]

        text = "\n".join(part for part in quoted_texts if part).strip()
        if self._is_unusable_quoted_text(text):
            text = ""
        if images and self._is_weak_image_caption_text(text):
            logger.info(
                f"[astrbot-fact-check-text-skip] weak image caption text={text!r}",
            )
            text = ""
        if inline_text and not self._is_unusable_quoted_text(inline_text):
            text = (text + "\n" + inline_text).strip() if text else inline_text

        images = self._dedupe_image_inputs(images)[:max_images]
        return FactCheckRequest(
            text=text[:3000],
            trigger_text=trigger_text,
            speaker=speaker,
            images=images,
        )

    async def _fetch_reply_payload(
        self,
        event: AstrMessageEvent,
        reply: Reply,
    ) -> tuple[list[str], list[Image], str] | None:
        """Fetch quoted message through AstrBot's OneBot quoted-message parser."""
        reply_id = str(getattr(reply, "id", "") or "").strip()
        if not reply_id:
            logger.info(
                f"[astrbot-fact-check-reply-fetch-skip] invalid reply id: {reply_id!r}"
            )
            return None

        try:
            text = await extract_quoted_message_text(event, reply)
            image_refs = await extract_quoted_message_images(event, reply)
        except Exception as exc:
            logger.warning(
                f"[astrbot-fact-check-reply-fetch-error] message_id={reply_id}: {exc!r}",
            )
            return None

        texts: list[str] = []
        images: list[Image] = []
        if text and not self._is_unusable_quoted_text(text):
            texts.append(text.strip())
        for ref in image_refs:
            ref = str(ref or "").strip()
            if not ref:
                continue
            images.append(
                Image(
                    file=ref, url=ref if ref.startswith(("http://", "https://")) else ""
                )
            )
        combined_text = "\n".join(part for part in texts if part).strip()
        logger.info(
            f"[astrbot-fact-check-reply-fetch] message_id={reply_id}: "
            f"text_len={len(combined_text)} images={len(images)}",
        )
        if not texts and not images:
            return None
        return texts, images, ""

    async def _append_forward_payloads(
        self,
        event: AstrMessageEvent,
        forward_ids: Iterable[str],
        *,
        quoted_texts: list[str],
        images: list[ImageInput],
        max_images: int,
        speaker: str,
        label: str,
        seen_image_refs: set[str],
    ) -> str:
        fetched = await self._fetch_forward_payloads(event, forward_ids, label=label)
        if not fetched:
            return speaker
        fetched_texts, fetched_images, fetched_speaker = fetched
        if fetched_speaker and not speaker:
            speaker = fetched_speaker
        for fetched_text in fetched_texts:
            if fetched_text and fetched_text not in quoted_texts:
                quoted_texts.append(fetched_text)
        images.extend(
            await self._image_inputs(
                fetched_images,
                remaining=max_images - len(images),
                seen_refs=seen_image_refs,
            ),
        )
        return speaker

    async def _fetch_forward_payloads(
        self,
        event: AstrMessageEvent,
        forward_ids: Iterable[str],
        *,
        label: str,
    ) -> tuple[list[str], list[Image], str] | None:
        ids = self._dedupe_forward_ids(forward_ids)
        if not ids:
            return None

        max_fetch = max(
            1, min(8, int(self.config.get("fact_check_forward_max_fetch") or 3))
        )
        parser = OneBotPayloadParser()
        client = OneBotClient(event)
        pending = list(ids)
        seen: set[str] = set()
        texts: list[str] = []
        image_refs: list[str] = []
        fetched_count = 0

        while pending and fetched_count < max_fetch:
            current_id = pending.pop(0)
            if current_id in seen:
                continue
            seen.add(current_id)
            fetched_count += 1
            try:
                payload = await client.get_forward_msg(current_id)
            except Exception as exc:
                logger.warning(
                    f"[astrbot-fact-check-forward-fetch-error] {label}: id={self._short_ref(current_id)} {exc!r}",
                )
                continue
            if not payload:
                logger.info(
                    f"[astrbot-fact-check-forward-fetch-empty] {label}: id={self._short_ref(current_id)}",
                )
                continue
            parsed = parser.parse_get_forward_payload(payload)
            parsed_text = self._clean_forward_text(str(parsed.get("text") or ""))
            if parsed_text:
                texts.append(parsed_text)
            for ref in parsed.get("image_refs") or []:
                ref_text = str(ref or "").strip()
                if ref_text:
                    image_refs.append(ref_text)
            for nested_id in parsed.get("forward_ids") or []:
                nested_text = str(nested_id or "").strip()
                if nested_text and nested_text not in seen:
                    pending.append(nested_text)

        if pending:
            logger.info(
                f"[astrbot-fact-check-forward-fetch-limit] {label}: "
                f"fetched={fetched_count} remaining={len(pending)}",
            )

        images: list[Image] = []
        for ref in self._dedupe_forward_ids(image_refs):
            images.append(
                Image(
                    file=ref, url=ref if ref.startswith(("http://", "https://")) else ""
                )
            )
        combined_text = "\n".join(texts).strip()
        logger.info(
            f"[astrbot-fact-check-forward-fetch] {label}: "
            f"roots={len(ids)} fetched={fetched_count} text_len={len(combined_text)} images={len(images)}",
        )
        if not texts and not images:
            return None
        return texts, images, ""

    def _extract_forward_ids_from_components(
        self, components: Iterable[object] | None
    ) -> list[str]:
        ids: list[str] = []
        if not components:
            return ids
        for comp in components:
            if isinstance(comp, Forward):
                ids.append(str(getattr(comp, "id", "") or "").strip())
            elif isinstance(comp, Plain):
                ids.extend(self._extract_forward_ids_from_text(str(comp.text or "")))
            elif isinstance(comp, Reply):
                ids.extend(
                    self._extract_forward_ids_from_text(str(comp.message_str or ""))
                )
                ids.extend(
                    self._extract_forward_ids_from_components(
                        getattr(comp, "chain", None)
                    )
                )
            elif isinstance(comp, Node):
                ids.extend(
                    self._extract_forward_ids_from_components(
                        getattr(comp, "content", None)
                    )
                )
            elif isinstance(comp, Nodes):
                for node in getattr(comp, "nodes", []) or []:
                    ids.extend(
                        self._extract_forward_ids_from_components(
                            getattr(node, "content", None)
                        )
                    )
        return self._dedupe_forward_ids(ids)

    def _extract_forward_ids_from_text(self, text: str | None) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        ids: list[str] = []
        for match in re.finditer(r"\[CQ:forward,([^\]]+)\]", text, flags=re.IGNORECASE):
            attrs = match.group(1)
            ids.extend(
                value.strip()
                for value in re.findall(
                    r"(?:^|,)(?:id|message_id|resid|m_resid|fileid|fid)=([^,\]\s]+)",
                    attrs,
                    flags=re.IGNORECASE,
                )
                if value.strip()
            )
        ids.extend(self._extract_multimsg_forward_ids(text))
        return self._dedupe_forward_ids(ids)

    def _extract_multimsg_forward_ids(self, text: str) -> list[str]:
        decoded = html.unescape(str(text or "")).replace("&#44;", ",")
        if "com.tencent.multimsg" not in decoded and "resid" not in decoded:
            return []

        ids: list[str] = []
        decoder = json.JSONDecoder()
        start = decoded.find("{")
        while start >= 0:
            try:
                value, end = decoder.raw_decode(decoded[start:])
            except json.JSONDecodeError:
                start = decoded.find("{", start + 1)
                continue
            if self._looks_like_multimsg_payload(value):
                ids.extend(self._walk_multimsg_forward_ids(value))
            start = decoded.find("{", start + max(end, 1))

        if not ids:
            ids.extend(
                value.strip()
                for value in re.findall(
                    r'"(?:resid|m_resid|forward_id|fileid|fid)"\s*:\s*"([^"]+)"',
                    decoded,
                    flags=re.IGNORECASE,
                )
                if value.strip()
            )
        return self._dedupe_forward_ids(ids)

    def _looks_like_multimsg_payload(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if value.get("app") == "com.tencent.multimsg":
            return True
        config = value.get("config")
        if isinstance(config, dict) and str(config.get("forward") or "") == "1":
            return True
        prompt = str(value.get("prompt") or value.get("desc") or "")
        return "聊天记录" in prompt or "合并转发" in prompt

    def _walk_multimsg_forward_ids(self, value: Any) -> list[str]:
        ids: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in {"resid", "m_resid", "forward_id", "fileid", "fid"}:
                    item_text = str(item or "").strip()
                    if item_text:
                        ids.append(item_text)
                else:
                    ids.extend(self._walk_multimsg_forward_ids(item))
        elif isinstance(value, list):
            for item in value:
                ids.extend(self._walk_multimsg_forward_ids(item))
        return self._dedupe_forward_ids(ids)

    @staticmethod
    def _dedupe_forward_ids(values: Iterable[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip().strip("\"'")
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    @staticmethod
    def _clean_forward_text(text: str) -> str:
        lines: list[str] = []
        for line in str(text or "").splitlines():
            cleaned = re.sub(
                r"\[(?:Image|Forward Message|Video)\]", " ", line, flags=re.IGNORECASE
            ).strip()
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines).strip()

    def _is_unusable_quoted_text(self, text: str | None) -> bool:
        if not isinstance(text, str):
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        if all(self._is_fact_check_command_only(line) for line in lines):
            return True
        placeholder_patterns = [
            r"^\[?(?:CQ:)?forward[,:\s].*(?:id|resid|m_resid|fileid|fid)=[^,\]\s]+.*\]?$",
            r"^\[CQ:json,.*com\.tencent\.multimsg.*\]$",
            r"^\[(?:合并转发|转发消息|forward message)[:：]?\d*\]$",
            r"^\[引用消息\]$",
            r"^\[Forward Message\]$",
            r"^\d{12,}$",
        ]
        return all(
            any(
                re.fullmatch(pattern, line, flags=re.IGNORECASE)
                for pattern in placeholder_patterns
            )
            for line in lines
        )

    @staticmethod
    def _is_fact_check_command_only(text: str | None) -> bool:
        if not isinstance(text, str):
            return False
        cleaned = text.strip()
        if not cleaned:
            return False
        cleaned = re.sub(r"\[?At[:：,][^\]]+\]?", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"@\S+", " ", cleaned)
        cleaned = cleaned.strip(" \t\r\n:：，,。.!！?？")
        if not cleaned:
            return False
        return bool(is_trigger(cleaned) and not remove_trigger(cleaned).strip())

    def _is_weak_image_caption_text(self, text: str | None) -> bool:
        if not isinstance(text, str):
            return False
        cleaned = remove_trigger(text).strip()
        if not cleaned:
            return True
        normalized = re.sub(
            r"[\s\u00a0\u200b\u200c\u200d:：，,。.!！?？/]+", "", cleaned
        )
        if not normalized:
            return True
        placeholder_values = {
            "图片",
            "[图片]",
            "图",
            "截图",
            "表情",
            "[表情]",
            "查看图片",
            "请核查",
            "事实核查",
            "核查",
        }
        if normalized in placeholder_values:
            return True
        if len(normalized) <= 8 and re.fullmatch(
            r"[\[\]【】()（）A-Za-z0-9_\-]+", normalized
        ):
            return True
        return False

    def _plain_texts(self, components: Iterable[object]) -> list[str]:
        texts: list[str] = []
        for comp in components:
            if isinstance(comp, Plain) and comp.text.strip():
                texts.append(comp.text.strip())
        return texts

    async def _image_inputs(
        self,
        components: Iterable[object],
        *,
        remaining: int,
        seen_refs: set[str] | None = None,
    ) -> list[ImageInput]:
        if remaining <= 0:
            return []
        images: list[ImageInput] = []
        seen_refs = seen_refs if seen_refs is not None else set()
        resolve_timeout = max(
            1.0,
            float(self.config.get("fact_check_image_download_timeout_seconds") or 10),
        )
        for comp in components:
            if not isinstance(comp, Image):
                continue
            file_name = str(comp.file or "").strip()
            original_path = str(getattr(comp, "path", "") or "").strip()
            url = str(comp.url or "").strip()
            if not url and is_public_http_url(file_name):
                url = file_name
            ref_key = self._image_component_ref_key(
                url=url,
                path=original_path,
                file_name=file_name,
            )
            log_label = safe_image_log_label(
                ImageInput(
                    url=url,
                    file_name=file_name,
                    path=original_path,
                ),
            )
            if ref_key and ref_key in seen_refs:
                logger.info(
                    f"[astrbot-fact-check-image-ref-dedupe] skipped {log_label}",
                )
                continue
            if ref_key:
                seen_refs.add(ref_key)

            path = original_path
            if path:
                path = self._snapshot_image_path(path, file_name=file_name)
                logger.info(
                    f"[astrbot-fact-check-image-local] {log_label} snapshot={'ok' if path else 'missing'}",
                )
            else:
                try:
                    path = await asyncio.wait_for(
                        comp.convert_to_file_path(), timeout=resolve_timeout
                    )
                    path = self._snapshot_image_path(path, file_name=file_name)
                    logger.info(
                        f"[astrbot-fact-check-image-local] {log_label} snapshot={'ok' if path else 'missing'}",
                    )
                except Exception as exc:
                    logger.warning(
                        f"[astrbot-fact-check-image-local-error] "
                        f"{log_label}: {type(exc).__name__}",
                    )
            if url and not is_public_http_url(url):
                logger.warning(
                    f"[astrbot-fact-check-image-skip] non-public {log_label}",
                )
                url = ""
            if not path and not url:
                continue
            content_sha256 = ""
            if path:
                content_sha256 = self._image_input_digest(
                    ImageInput(url=url, file_name=file_name, path=path)
                )
            images.append(
                ImageInput(
                    url=url,
                    file_name=file_name,
                    path=path,
                    content_sha256=content_sha256,
                ),
            )
            if len(images) >= remaining:
                break
        return images

    @staticmethod
    def _image_component_ref_key(
        *,
        url: str,
        path: str,
        file_name: str,
    ) -> str:
        remote = str(url or "").strip()
        if remote:
            return "url:" + remote.split("#", 1)[0]
        local = str(path or "").strip()
        if not local and file_name and not is_public_http_url(file_name):
            local = str(file_name).strip()
        if not local:
            return ""
        return "local:" + local.replace("\\", "/").casefold()

    def _snapshot_image_path(self, path_value: str, *, file_name: str = "") -> str:
        source = Path(
            str(path_value or "").removeprefix("file:///").removeprefix("file://")
        )
        if not source.is_file():
            return str(path_value or "")
        try:
            cache_dir = Path(StarTools.get_data_dir()) / "input_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._prune_image_input_cache(cache_dir)
            suffix = source.suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                suffix = Path(str(file_name or "").split("?", 1)[0]).suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                suffix = ".img"
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            target = cache_dir / f"{digest.hexdigest()}{suffix}"
            if not target.exists():
                shutil.copy2(source, target)
            target.touch()
            self._prune_image_input_cache(
                cache_dir,
                force=True,
                protected=target,
            )
            return str(target)
        except Exception as exc:
            logger.warning(
                f"[astrbot-fact-check-image-snapshot-error] "
                f"local:{source.name or 'image'}: {type(exc).__name__}",
            )
            return str(path_value or "")

    def _prune_image_input_cache(
        self,
        cache_dir: Path,
        *,
        force: bool = False,
        protected: Path | None = None,
    ) -> None:
        now = time.time()
        last_prune = float(getattr(self, "_image_cache_last_prune", 0.0) or 0.0)
        if not force and now - last_prune < 300:
            return
        self._image_cache_last_prune = now
        default_ttl = max(
            7200,
            int(self.config.get("fact_check_followup_ttl_seconds") or 3600) + 600,
        )
        max_age = max(
            60,
            int(self.config.get("fact_check_image_cache_ttl_seconds") or default_ttl),
        )
        max_bytes = max(
            1,
            int(
                self.config.get("fact_check_image_cache_max_bytes") or 64 * 1024 * 1024
            ),
        )
        max_files = max(
            1,
            int(self.config.get("fact_check_image_cache_max_files") or 100),
        )
        protected_path = protected.resolve() if protected else None
        entries: list[tuple[Path, float, int]] = []
        for item in cache_dir.iterdir():
            try:
                if not item.is_file():
                    continue
                stat = item.stat()
                if now - stat.st_mtime > max_age and (
                    protected_path is None or item.resolve() != protected_path
                ):
                    item.unlink()
                    continue
                entries.append((item, stat.st_mtime, stat.st_size))
            except OSError:
                continue

        total_bytes = sum(size for _, _, size in entries)
        entries.sort(key=lambda entry: entry[1])
        while len(entries) > max_files or total_bytes > max_bytes:
            removable_index = next(
                (
                    index
                    for index, (item, _, _) in enumerate(entries)
                    if protected_path is None or item.resolve() != protected_path
                ),
                None,
            )
            if removable_index is None:
                break
            item, _, size = entries.pop(removable_index)
            try:
                item.unlink()
                total_bytes -= size
            except OSError:
                continue

    def _failed_fact_check_reply(self, reason: str) -> str:
        if bool(self.config.get("fact_check_show_failure_reason", True)):
            return explain_failure(reason)
        return FAILED_REPLY

    def _event_label(self, event: AstrMessageEvent) -> str:
        group_id = str(event.get_group_id() or "").strip()
        user_id = str(event.get_sender_id() or "").strip()
        if group_id:
            return f"group:{group_id}:user:{user_id}"
        return f"private:{user_id}"

    def _list_config(self, key: str, default: list[str]) -> list[str]:
        value = self.config.get(key, default)
        if isinstance(value, str):
            items = value.split(",")
        else:
            try:
                items = list(value)
            except TypeError:
                items = list(default)
        return [str(item).strip() for item in items if str(item).strip()]

    def _short_ref(self, value: str, limit: int = 120) -> str:
        value = str(value or "").replace("\n", " ").strip()
        return value if len(value) <= limit else value[:limit] + "..."

    async def terminate(self):
        self._persist_fact_check_sessions()
        current_task = asyncio.current_task()
        tasks = [
            task
            for task in (*self._fact_check_tasks, *self._followup_tasks)
            if task is not current_task
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._singleflight.cancel_all()
        logger.info("[astrbot-fact-check] terminated")
