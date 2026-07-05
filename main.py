from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Forward, Image, Node, Nodes, Plain, Reply
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.utils.quoted_message.extractor import (
    extract_quoted_message_images,
    extract_quoted_message_text,
)
from astrbot.core.utils.quoted_message.chain_parser import OneBotPayloadParser
from astrbot.core.utils.quoted_message.onebot_client import OneBotClient
from astrbot.core.star.filter.custom_filter import CustomFilter

from .fact_check import (
    FAILED_REPLY,
    FactCheckRequest,
    FactCheckResult,
    ImageInput,
    explain_failure,
    is_public_http_url,
    is_trigger,
    remove_trigger,
    run_fact_check,
    run_fact_check_followup,
)

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
    from astrbot_plugin_qq_agent_core.media_send import is_confirm_timeout, send_chain_result
except Exception:
    try:
        plugins_dir = Path(__file__).resolve().parents[1]
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        from astrbot_plugin_qq_agent_core.media_send import is_confirm_timeout, send_chain_result
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
        self._active_followup_jobs = 0
        self._reply_cache: dict[str, tuple[float, FactCheckResult]] = {}
        self._fact_check_sessions: dict[str, FactCheckSession] = {}
        self._cooldown_until = 0.0

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
                    event.plain_result("这条事实核查上下文已过期，请重新发送 /事实核查 再查一次。")
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
            await event.send(event.plain_result(f"Gemini 刚被限流，先冷却 {int(cooldown_left) + 1} 秒。"))
            return
        if self._fact_check_queue_full():
            logger.warning(
                f"[astrbot-fact-check-followup-queue-full] {label}: "
                f"jobs={self._active_fact_check_jobs()} max={self._max_fact_check_queue()}",
            )
            await event.send(event.plain_result("事实核查队列满了，等前面的跑完再试一下。"))
            return

        await event.send(event.plain_result("我接着查一下。"))
        self._active_followup_jobs = max(0, int(getattr(self, "_active_followup_jobs", 0))) + 1
        try:
            async with self._fact_check_semaphore:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
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
                            str(model).strip()
                            for model in self.config.get("fact_check_main_models", [])
                            if str(model).strip()
                        ]
                        or [
                            "gemini-3-flash-preview",
                            "gemini-2.5-flash",
                            "gemini-3.1-flash-lite",
                        ],
                        request_timeout=int(self.config.get("fact_check_main_timeout_seconds") or 45),
                    ),
                    timeout=max(10.0, float(self.config.get("fact_check_total_timeout_seconds") or 90)),
                )
        except asyncio.TimeoutError:
            reason = f"follow-up timeout after {time.perf_counter() - started_at:.1f}s"
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
            self._active_followup_jobs = max(0, int(getattr(self, "_active_followup_jobs", 0)) - 1)

        logger.info(
            f"[astrbot-fact-check-followup-done] {label}: "
            f"session={session.session_id} {time.perf_counter() - started_at:.2f}s",
        )
        await self._send_fact_check_reply(
            event,
            result.reply or FAILED_REPLY,
            label=label,
            purpose="followup",
            session_id=session.session_id,
        )

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

        request_data = await self._build_fact_check_request(event, trigger_text=trigger_text)
        if not request_data.text and not request_data.images:
            if self._is_fact_check_command_only(trigger_text):
                yield event.plain_result(self._fact_check_usage_text())
                return
            reason = "no quoted text or inline claim"
            logger.info(f"[astrbot-fact-check-reason] {self._event_label(event)}: {reason}")
            yield event.plain_result(self._failed_fact_check_reply(reason))
            return

        cache_key = self._request_cache_key(request_data)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"[astrbot-fact-check-cache-hit] {self._event_label(event)}: key={cache_key[:12]}")
            session_id = self._remember_fact_check_session(event, request_data, cached_result)
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
            yield event.plain_result(f"Gemini 刚被限流，先冷却 {int(cooldown_left) + 1} 秒。")
            return

        if self._fact_check_queue_full():
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
        await event.send(event.plain_result("我先查一下。"))
        task = asyncio.create_task(self._run_fact_check_job(event, request_data, started_at, cache_key))
        self._fact_check_tasks.add(task)
        task.add_done_callback(self._fact_check_tasks.discard)
        return

    async def _run_fact_check_job(
        self,
        event: AstrMessageEvent,
        request_data: FactCheckRequest,
        started_at: float,
        cache_key: str,
    ) -> None:
        label = self._event_label(event)
        try:
            async with self._fact_check_semaphore:
                timeout_seconds = max(
                    10.0,
                    float(self.config.get("fact_check_total_timeout_seconds") or 90),
                )
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_fact_check,
                        request_data=request_data,
                        api_key=str(self.config.get("gemini_api_key") or ""),
                        base_url=str(
                            self.config.get(
                                "gemini_base_url",
                                "https://generativelanguage.googleapis.com/v1beta/models",
                            )
                            or "https://generativelanguage.googleapis.com/v1beta/models",
                        ),
                        pre_model=str(self.config.get("fact_check_pre_model") or "gemini-3.1-flash-lite"),
                        main_models=[
                            str(model).strip()
                            for model in self.config.get("fact_check_main_models", [])
                            if str(model).strip()
                        ]
                        or [
                            "gemini-3-flash-preview",
                            "gemini-2.5-flash",
                            "gemini-3.1-flash-lite",
                        ],
                        max_image_bytes=int(self.config.get("fact_check_max_image_bytes") or 5 * 1024 * 1024),
                        long_image_chunk_height=int(
                            self.config.get("fact_check_long_image_chunk_height") or 2200,
                        ),
                        long_image_max_parts=int(
                            self.config.get("fact_check_long_image_max_parts") or 8,
                        ),
                        long_image_max_width=int(
                            self.config.get("fact_check_long_image_max_width") or 1280,
                        ),
                        image_download_timeout=int(
                            self.config.get("fact_check_image_download_timeout_seconds") or 10,
                        ),
                        pre_request_timeout=int(
                            self.config.get("fact_check_pre_timeout_seconds") or 25,
                        ),
                        main_request_timeout=int(
                            self.config.get("fact_check_main_timeout_seconds") or 45,
                        ),
                        anysearch_enabled=bool(
                            self.config.get("fact_check_anysearch_enabled", False),
                        ),
                        anysearch_endpoint=str(
                            self.config.get("fact_check_anysearch_endpoint")
                            or "https://api.anysearch.com/mcp",
                        ),
                        anysearch_api_key=str(
                            self.config.get("fact_check_anysearch_api_key") or "",
                        ),
                        anysearch_timeout=int(
                            self.config.get("fact_check_anysearch_timeout_seconds") or 20,
                        ),
                        anysearch_max_claims=int(
                            self.config.get("fact_check_anysearch_max_claims") or 3,
                        ),
                        anysearch_max_results_per_claim=int(
                            self.config.get("fact_check_anysearch_max_results_per_claim") or 3,
                        ),
                        anysearch_extract_top_urls=int(
                            self.config.get("fact_check_anysearch_extract_top_urls") or 2,
                        ),
                        anysearch_max_chars=int(
                            self.config.get("fact_check_anysearch_max_chars") or 6000,
                        ),
                        anysearch_freshness=str(
                            self.config.get("fact_check_anysearch_freshness") or "",
                        ),
                        anysearch_content_types=self._list_config(
                            "fact_check_anysearch_content_types",
                            ["web", "news"],
                        ),
                    ),
                    timeout=timeout_seconds,
                )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - started_at
            reason = f"timeout after {elapsed:.1f}s"
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
        self._set_cached_result(cache_key, result)
        session_id = self._remember_fact_check_session(event, request_data, result)
        await self._send_fact_check_reply(
            event,
            result.reply or FAILED_REPLY,
            label=label,
            purpose="result",
            session_id=session_id,
        )

    async def _send_fact_check_reply(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        label: str,
        purpose: str,
        session_id: str | None = None,
    ) -> None:
        text = str(text or FAILED_REPLY).strip() or FAILED_REPLY
        logger.info(
            f"[astrbot-fact-check-send] {label}: purpose={purpose} len={len(text)}",
        )
        forward_result = self._fact_check_forward_result(event, text, session_id=session_id)
        safe_text = self._sanitize_forward_text_for_qq(text)
        retry_result = (
            self._fact_check_forward_result(event, safe_text, session_id=session_id)
            if safe_text != text
            else forward_result
        )

        outcome = await send_chain_result(event, forward_result) if send_chain_result else None
        try:
            if outcome is None:
                await event.send(forward_result)
            elif outcome.assumed_ok:
                logger.info(
                    f"[astrbot-fact-check-send-assume-ok] {label}: "
                    f"method=event.send.forward kind={outcome.kind} error={outcome.error}",
                )
                return
            elif not outcome.ok:
                raise RuntimeError(f"{outcome.kind}: {outcome.error}")
            logger.info(f"[astrbot-fact-check-send-ok] {label}: method=event.send.forward")
            return
        except Exception as exc:
            if self._looks_like_confirm_timeout(exc):
                logger.info(
                    f"[astrbot-fact-check-send-assume-ok] {label}: "
                    f"method=event.send.forward error={exc!r}",
                )
                return
            logger.warning(
                f"[astrbot-fact-check-send-error] {label}: "
                f"method=event.send.forward kind={getattr(outcome, 'kind', None) or 'unknown'} error={exc!r}",
            )

        await asyncio.sleep(1.0)
        outcome = await send_chain_result(event, retry_result) if send_chain_result else None
        try:
            if safe_text != text:
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
                return
            elif not outcome.ok:
                raise RuntimeError(f"{outcome.kind}: {outcome.error}")
            logger.info(
                f"[astrbot-fact-check-send-ok] {label}: method=event.send.forward retry"
            )
            return
        except Exception as exc:
            if self._looks_like_confirm_timeout(exc):
                logger.info(
                    f"[astrbot-fact-check-send-assume-ok] {label}: "
                    f"method=event.send.forward retry error={exc!r}",
                )
                return
            logger.error(
                f"[astrbot-fact-check-send-failed] {label}: "
                f"method=event.send.forward retry kind={getattr(outcome, 'kind', None) or 'unknown'} error={exc!r}",
            )
            self._dump_forward_failure(label, text, exc)
            fallback_text = self._fact_check_text_with_session_marker(safe_text, session_id=session_id)
            if await self._send_text_via_onebot(
                event,
                fallback_text,
                label=label,
                prefer_send_msg=True,
                suppress_errors=True,
            ):
                logger.info(f"[astrbot-fact-check-send-ok] {label}: method=onebot fallback")
                return
            try:
                await event.send(event.plain_result(fallback_text))
                logger.info(f"[astrbot-fact-check-send-ok] {label}: method=plain fallback")
            except Exception as fallback_exc:
                logger.error(
                    f"[astrbot-fact-check-send-failed] {label}: "
                    f"method=plain fallback error={fallback_exc!r}",
                )

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
            logger.warning(f"[astrbot-fact-check-send-skip] {label}: no onebot call_action")
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
            logger.warning(f"[astrbot-fact-check-send-skip] {label}: invalid target={target_value!r}")
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
                    logger.info(
                        f"[astrbot-fact-check-send-assume-ok] {label}: "
                        f"method=onebot action={action} chunks={len(chunks)} "
                        f"sent={next_index} attempt={attempt} error={exc!r}",
                    )
                    return True
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
        replacements = {
            "翻墙": "翻 墙",
            "网信办": "网 信 办",
            "国家网络安全法规": "相关网络安全法规",
            "违规访问": "不合规访问",
            "Reddit": "R eddit",
            "r/China_irl": "r / China_irl",
            "大纪元": "大 纪 元",
            "VPN": "V PN",
            "vpn": "v pn",
        }
        safe = str(text or "")
        for source, target in replacements.items():
            safe = safe.replace(source, target)
        return safe

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
    def _fact_check_text_with_session_marker(text: str, *, session_id: str | None) -> str:
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
                "text": text,
                "length": len(text),
                "chunks": chunks,
                "chunk_lengths": [len(chunk) for chunk in chunks],
                "error": repr(exc),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[astrbot-fact-check-forward-failure-dump] {label}: {path}")
        except Exception as dump_exc:
            logger.warning(f"[astrbot-fact-check-forward-failure-dump-error] {label}: {dump_exc!r}")

    def _request_cache_key(self, request_data: FactCheckRequest) -> str:
        def cache_config_value(key: str, default):
            value = self.config.get(key, None)
            if value is None or value == "":
                return default
            return value

        anysearch_api_key = str(self.config.get("fact_check_anysearch_api_key") or "").strip()
        payload = {
            "text": request_data.text.strip(),
            "speaker": request_data.speaker.strip(),
            "images": [
                {
                    "url": image.url,
                    "file_name": image.file_name,
                    "path": image.path,
                }
                for image in request_data.images
            ],
            "anysearch": {
                "enabled": bool(self.config.get("fact_check_anysearch_enabled", False)),
                "endpoint": str(
                    self.config.get("fact_check_anysearch_endpoint") or "https://api.anysearch.com/mcp",
                ).strip(),
                "api_key_sha256": hashlib.sha256(anysearch_api_key.encode("utf-8")).hexdigest()
                if anysearch_api_key
                else "",
                "timeout_seconds": str(cache_config_value("fact_check_anysearch_timeout_seconds", 20)),
                "max_claims": str(cache_config_value("fact_check_anysearch_max_claims", 3)),
                "max_results_per_claim": str(
                    cache_config_value("fact_check_anysearch_max_results_per_claim", 3),
                ),
                "extract_top_urls": str(cache_config_value("fact_check_anysearch_extract_top_urls", 2)),
                "max_chars": str(cache_config_value("fact_check_anysearch_max_chars", 6000)),
                "freshness": str(self.config.get("fact_check_anysearch_freshness") or "").strip(),
                "content_types": self._list_config(
                    "fact_check_anysearch_content_types",
                    ["web", "news"],
                ),
            },
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_fact_check_allowed(self, event: AstrMessageEvent) -> bool:
        if is_plugin_allowed is None:
            return True
        return bool(
            is_plugin_allowed(
                "fact_check",
                event,
                default_allow=True,
                default_allow_private=True,
            )
        )

    def _max_fact_check_queue(self) -> int:
        return max(1, int(self.config.get("fact_check_max_queue") or 4))

    def _active_fact_check_jobs(self) -> int:
        return len(getattr(self, "_fact_check_tasks", set()) or set()) + max(
            0,
            int(getattr(self, "_active_followup_jobs", 0)),
        )

    def _fact_check_queue_full(self) -> bool:
        return self._active_fact_check_jobs() >= self._max_fact_check_queue()

    def _remember_fact_check_session(
        self,
        event: AstrMessageEvent,
        request_data: FactCheckRequest,
        result: FactCheckResult,
    ) -> str:
        self._cleanup_fact_check_sessions()
        session_id = "fc_" + uuid.uuid4().hex[:8]
        self._fact_check_sessions[session_id] = FactCheckSession(
            session_id=session_id,
            created_at=time.time(),
            group_id=str(event.get_group_id() or "").strip(),
            user_id=str(event.get_sender_id() or "").strip(),
            request_data=request_data,
            reply=result.reply or FAILED_REPLY,
            candidates=list(result.candidates or []),
            sources=list(result.sources or []),
        )
        self._cleanup_fact_check_sessions()
        logger.info(f"[astrbot-fact-check-session-save] {self._event_label(event)}: session={session_id}")
        return session_id

    def _cleanup_fact_check_sessions(self) -> None:
        ttl = max(60, int(self.config.get("fact_check_followup_ttl_seconds") or 3600))
        max_entries = max(8, int(self.config.get("fact_check_followup_max_sessions") or 50))
        now = time.time()
        expired = [
            session_id
            for session_id, session in self._fact_check_sessions.items()
            if now - session.created_at > ttl
        ]
        for session_id in expired:
            self._fact_check_sessions.pop(session_id, None)
        if len(self._fact_check_sessions) <= max_entries:
            return
        stale = sorted(self._fact_check_sessions.values(), key=lambda item: item.created_at)
        for session in stale[: len(self._fact_check_sessions) - max_entries]:
            self._fact_check_sessions.pop(session.session_id, None)

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

    async def _find_followup_session(self, event: AstrMessageEvent) -> FactCheckSession | None:
        session, _ = await self._find_followup_session_with_state(event)
        return session

    async def _find_followup_session_with_state(
        self,
        event: AstrMessageEvent,
    ) -> tuple[FactCheckSession | None, bool]:
        self._cleanup_fact_check_sessions()
        ids: list[str] = []
        quoted_looks_like_fact_check = False
        for text in _event_text_candidates(event):
            ids.extend(self._extract_fact_check_session_ids(text))

        for comp in event.get_messages():
            if not isinstance(comp, Reply):
                continue
            for text in [str(comp.message_str or "")] + self._plain_texts(comp.chain or []):
                ids.extend(self._extract_fact_check_session_ids(text))
                quoted_looks_like_fact_check = quoted_looks_like_fact_check or self._looks_like_fact_check_reply(text)
            fetched = await self._fetch_reply_payload(event, comp)
            if fetched:
                fetched_texts, _, _ = fetched
                for text in fetched_texts:
                    ids.extend(self._extract_fact_check_session_ids(text))
                    quoted_looks_like_fact_check = quoted_looks_like_fact_check or self._looks_like_fact_check_reply(text)

        for session_id in ids:
            session = self._fact_check_sessions.get(session_id)
            if session:
                return session, False

        # NapCat sometimes exposes a fact-check quote without the marker. Only then use latest session.
        if not quoted_looks_like_fact_check:
            return None, bool(ids)
        candidates = [
            session
            for session in self._fact_check_sessions.values()
            if self._session_visible_to_event(session, event)
        ]
        if not candidates:
            return None, True
        latest = candidates[0]
        for candidate in candidates[1:]:
            if candidate.created_at > latest.created_at:
                latest = candidate
        return latest, False
        return max(candidates, key=lambda item: item.created_at)

    @staticmethod
    def _extract_fact_check_session_ids(text: str | None) -> list[str]:
        if not text:
            return []
        return [match.group(0).lower() for match in re.finditer(r"fc_[0-9a-fA-F]{8,16}", str(text))]

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
        if "事实核查" in normalized and any(marker in normalized for marker in ("要点：", "来源：", "总结论")):
            return True
        return False

    @staticmethod
    def _session_visible_to_event(session: FactCheckSession, event: AstrMessageEvent) -> bool:
        group_id = str(event.get_group_id() or "").strip()
        if session.group_id:
            return bool(group_id and group_id == session.group_id)
        return str(event.get_sender_id() or "").strip() == session.user_id

    def _get_cached_result(self, cache_key: str) -> FactCheckResult | None:
        ttl = max(0, int(self.config.get("fact_check_cache_ttl_seconds") or 600))
        if ttl <= 0:
            return None
        cached = self._reply_cache.get(cache_key)
        if not cached:
            return None
        created_at, value = cached
        if time.time() - created_at > ttl:
            self._reply_cache.pop(cache_key, None)
            return None
        if isinstance(value, FactCheckResult):
            return value
        return FactCheckResult(str(value or FAILED_REPLY), "ok; cache", [], [])

    def _set_cached_result(self, cache_key: str, result: FactCheckResult) -> None:
        ttl = max(0, int(self.config.get("fact_check_cache_ttl_seconds") or 600))
        if ttl <= 0:
            return
        self._reply_cache[cache_key] = (
            time.time(),
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
            for key, (created_at, _) in self._reply_cache.items():
                if created_at < oldest_at:
                    oldest_key = key
                    oldest_at = created_at
            if not oldest_key:
                break
            self._reply_cache.pop(oldest_key, None)

    def _get_cached_reply(self, cache_key: str) -> str:
        result = self._get_cached_result(cache_key)
        return result.reply if result else ""

        ttl = max(0, int(self.config.get("fact_check_cache_ttl_seconds") or 600))
        if ttl <= 0:
            return ""
        cached = self._reply_cache.get(cache_key)
        if not cached:
            return ""
        created_at, reply = cached
        if time.time() - created_at > ttl:
            self._reply_cache.pop(cache_key, None)
            return ""
        return reply

    def _set_cached_reply(self, cache_key: str, reply: str) -> None:
        self._set_cached_result(cache_key, FactCheckResult(str(reply or FAILED_REPLY), "ok; cache", [], []))
        return

        ttl = max(0, int(self.config.get("fact_check_cache_ttl_seconds") or 600))
        if ttl <= 0:
            return
        self._reply_cache[cache_key] = (time.time(), reply)
        max_entries = max(8, int(self.config.get("fact_check_cache_max_entries") or 32))
        if len(self._reply_cache) <= max_entries:
            return
        stale = sorted(self._reply_cache.items(), key=lambda item: item[1][0])
        for key, _ in stale[: len(self._reply_cache) - max_entries]:
            self._reply_cache.pop(key, None)

    def _cooldown_left(self) -> float:
        return max(0.0, self._cooldown_until - time.time())

    def _maybe_start_cooldown(self, reason: str) -> None:
        lowered = str(reason or "").lower()
        if "429" not in lowered and "too many requests" not in lowered and "rate limit" not in lowered:
            return
        seconds = max(0, int(self.config.get("fact_check_rate_limit_cooldown_seconds") or 90))
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

    @staticmethod
    def _fact_check_usage_text() -> str:
        return "用法：回复一条消息后发送 /事实核查，或者直接发送 /事实核查 要核查的内容。"

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
        max_images = max(0, int(self.config.get("fact_check_max_images") or 2))

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
                    local_forward_ids.extend(self._extract_forward_ids_from_text(comp_text))
                if comp.chain:
                    local_texts.extend(self._plain_texts(comp.chain))
                    local_forward_ids.extend(self._extract_forward_ids_from_components(comp.chain))
                    images.extend(await self._image_inputs(comp.chain, remaining=max_images - len(images)))
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
                        images.extend(await self._image_inputs(fetched_images, remaining=max_images - len(images)))
                if local_forward_ids:
                    speaker = await self._append_forward_payloads(
                        event,
                        local_forward_ids,
                        quoted_texts=quoted_texts,
                        images=images,
                        max_images=max_images,
                        speaker=speaker,
                        label=f"reply:{getattr(comp, 'id', '')}",
                    )
            elif isinstance(comp, Forward):
                speaker = await self._append_forward_payloads(
                    event,
                    [str(getattr(comp, "id", "") or "")],
                    quoted_texts=quoted_texts,
                    images=images,
                    max_images=max_images,
                    speaker=speaker,
                    label="direct",
                )
            elif isinstance(comp, Plain):
                forward_ids = self._extract_forward_ids_from_text(str(comp.text or ""))
                if forward_ids:
                    speaker = await self._append_forward_payloads(
                        event,
                        forward_ids,
                        quoted_texts=quoted_texts,
                        images=images,
                        max_images=max_images,
                        speaker=speaker,
                        label="plain",
                    )
            elif isinstance(comp, Image):
                images.extend(await self._image_inputs([comp], remaining=max_images - len(images)))

            if len(images) >= max_images:
                images = images[:max_images]

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

        return FactCheckRequest(
            text=text[:3000],
            trigger_text=trigger_text,
            speaker=speaker,
            images=images[:max_images],
        )

    async def _fetch_reply_payload(
        self,
        event: AstrMessageEvent,
        reply: Reply,
    ) -> tuple[list[str], list[Image], str] | None:
        """Fetch quoted message through AstrBot's OneBot quoted-message parser."""
        reply_id = str(getattr(reply, "id", "") or "").strip()
        if not reply_id:
            logger.info(f"[astrbot-fact-check-reply-fetch-skip] invalid reply id: {reply_id!r}")
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
            images.append(Image(file=ref, url=ref if ref.startswith(("http://", "https://")) else ""))
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
        images.extend(await self._image_inputs(fetched_images, remaining=max_images - len(images)))
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

        max_fetch = max(1, min(8, int(self.config.get("fact_check_forward_max_fetch") or 3)))
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
            images.append(Image(file=ref, url=ref if ref.startswith(("http://", "https://")) else ""))
        combined_text = "\n".join(texts).strip()
        logger.info(
            f"[astrbot-fact-check-forward-fetch] {label}: "
            f"roots={len(ids)} fetched={fetched_count} text_len={len(combined_text)} images={len(images)}",
        )
        if not texts and not images:
            return None
        return texts, images, ""

    def _extract_forward_ids_from_components(self, components: Iterable[object] | None) -> list[str]:
        ids: list[str] = []
        if not components:
            return ids
        for comp in components:
            if isinstance(comp, Forward):
                ids.append(str(getattr(comp, "id", "") or "").strip())
            elif isinstance(comp, Plain):
                ids.extend(self._extract_forward_ids_from_text(str(comp.text or "")))
            elif isinstance(comp, Reply):
                ids.extend(self._extract_forward_ids_from_text(str(comp.message_str or "")))
                ids.extend(self._extract_forward_ids_from_components(getattr(comp, "chain", None)))
            elif isinstance(comp, Node):
                ids.extend(self._extract_forward_ids_from_components(getattr(comp, "content", None)))
            elif isinstance(comp, Nodes):
                for node in getattr(comp, "nodes", []) or []:
                    ids.extend(self._extract_forward_ids_from_components(getattr(node, "content", None)))
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
            cleaned = re.sub(r"\[(?:Image|Forward Message|Video)\]", " ", line, flags=re.IGNORECASE).strip()
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
            any(re.fullmatch(pattern, line, flags=re.IGNORECASE) for pattern in placeholder_patterns)
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
        normalized = re.sub(r"[\s\u00a0\u200b\u200c\u200d:：，,。.!！?？/]+", "", cleaned)
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
        if len(normalized) <= 8 and re.fullmatch(r"[\[\]【】()（）A-Za-z0-9_\-]+", normalized):
            return True
        return False

    def _plain_texts(self, components: Iterable[object]) -> list[str]:
        texts: list[str] = []
        for comp in components:
            if isinstance(comp, Plain) and comp.text.strip():
                texts.append(comp.text.strip())
        return texts

    async def _image_inputs(self, components: Iterable[object], *, remaining: int) -> list[ImageInput]:
        if remaining <= 0:
            return []
        images: list[ImageInput] = []
        resolve_timeout = max(1.0, float(self.config.get("fact_check_image_download_timeout_seconds") or 10))
        for comp in components:
            if not isinstance(comp, Image):
                continue
            file_name = str(comp.file or "").strip()
            path = str(getattr(comp, "path", "") or "").strip()
            if not path:
                try:
                    path = await asyncio.wait_for(comp.convert_to_file_path(), timeout=resolve_timeout)
                    logger.info(
                        f"[astrbot-fact-check-image-local] {self._short_ref(file_name or str(comp.url or ''))}: {path}",
                    )
                except Exception as exc:
                    logger.warning(
                        f"[astrbot-fact-check-image-local-error] "
                        f"{self._short_ref(file_name or str(comp.url or ''))}: {exc!r}",
                    )
            url = str(comp.url or "").strip()
            if not url and is_public_http_url(file_name):
                url = file_name
            if url and not is_public_http_url(url):
                logger.warning(
                    f"[astrbot-fact-check-image-skip] non-public url={self._short_ref(url)}",
                )
                url = ""
            if not path and not url:
                continue
            images.append(ImageInput(url=url, file_name=file_name, path=path))
            if len(images) >= remaining:
                break
        return images

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
        tasks = list(self._fact_check_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("[astrbot-fact-check] terminated")
