from __future__ import annotations

import asyncio
import re
import time
from typing import Iterable

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Node, Nodes, Plain, Reply
from astrbot.api.star import Context, Star
from astrbot.core.utils.quoted_message.extractor import (
    extract_quoted_message_images,
    extract_quoted_message_text,
)
from astrbot.core.star.filter.custom_filter import CustomFilter

from .fact_check import (
    FAILED_REPLY,
    FactCheckRequest,
    ImageInput,
    explain_failure,
    is_trigger,
    remove_trigger,
    run_fact_check,
)


class FactCheckWakeFilter(CustomFilter):
    """Wake only for explicit fact-check triggers."""

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        if event.is_private_chat():
            return is_trigger(event.message_str)
        return bool(
            getattr(event, "is_at_or_wake_command", False)
            or is_trigger(event.message_str)
        )


class FactCheckPlugin(Star):
    """Standalone QQ-friendly fact-check command."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

    @filter.custom_filter(FactCheckWakeFilter, priority=998_000)
    async def fact_check(self, event: AstrMessageEvent):
        """Run fact-checking when users say /事实核查, factcheck, or fact-check."""
        if not bool(self.config.get("enable_fact_check", True)):
            return
        if not is_trigger(event.message_str):
            return

        started_at = time.perf_counter()
        event.set_extra("qq_agent_command_handled", True)
        event.stop_event()
        request_data = await self._build_fact_check_request(event)
        if not request_data.text and not request_data.images:
            reason = "no quoted text or inline claim"
            logger.info(f"[astrbot-fact-check-reason] {self._event_label(event)}: {reason}")
            yield event.plain_result(self._failed_fact_check_reply(reason))
            return

        logger.info(
            f"[astrbot-fact-check-queue] {self._event_label(event)}: "
            f"text_len={len(request_data.text)} images={len(request_data.images)}",
        )
        await event.send(event.plain_result("我先查一下。"))
        asyncio.create_task(self._run_fact_check_job(event, request_data, started_at))
        return

    async def _run_fact_check_job(
        self,
        event: AstrMessageEvent,
        request_data: FactCheckRequest,
        started_at: float,
    ) -> None:
        label = self._event_label(event)
        try:
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
                    pre_model=str(self.config.get("fact_check_pre_model") or "gemini-3.1-flash-lite-preview"),
                    main_models=[
                        str(model).strip()
                        for model in self.config.get("fact_check_main_models", [])
                        if str(model).strip()
                    ]
                    or [
                        "gemini-3-flash-preview",
                        "gemini-2.5-flash",
                        "gemini-3.1-flash-lite-preview",
                    ],
                    max_image_bytes=int(self.config.get("fact_check_max_image_bytes") or 2 * 1024 * 1024),
                    image_download_timeout=int(
                        self.config.get("fact_check_image_download_timeout_seconds") or 10,
                    ),
                    pre_request_timeout=int(
                        self.config.get("fact_check_pre_timeout_seconds") or 25,
                    ),
                    main_request_timeout=int(
                        self.config.get("fact_check_main_timeout_seconds") or 45,
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
        await self._send_fact_check_reply(
            event,
            result.reply or FAILED_REPLY,
            label=label,
            purpose="result",
        )

    async def _send_fact_check_reply(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        label: str,
        purpose: str,
    ) -> None:
        text = str(text or FAILED_REPLY).strip() or FAILED_REPLY
        logger.info(
            f"[astrbot-fact-check-send] {label}: purpose={purpose} len={len(text)}",
        )
        forward_result = self._fact_check_forward_result(event, text)

        try:
            await event.send(forward_result)
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
                f"method=event.send.forward error={exc!r}",
            )

        await asyncio.sleep(1.0)
        try:
            await event.send(forward_result)
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
                f"method=event.send.forward retry error={exc!r}",
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
        for attempt in range(1, 4):
            try:
                for index, chunk in enumerate(chunks, start=1):
                    payload = {
                        target_key: int(target_value),
                        "message": [{"type": "text", "data": {"text": chunk}}],
                    }
                    if action == "send_msg":
                        payload["message_type"] = "group"
                    await call_action(action, **payload)
                    if index < len(chunks):
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
                        f"attempt={attempt} error={exc!r}",
                    )
                    return True
                message = (
                    f"[astrbot-fact-check-send-error] {label}: "
                    f"method=onebot action={action} chunks={len(chunks)} "
                    f"attempt={attempt}/3 error={exc!r}"
                )
                if suppress_errors:
                    logger.info(message)
                else:
                    logger.warning(message)
                await asyncio.sleep(1.0 * attempt)
        return False

    def _looks_like_confirm_timeout(self, exc: Exception) -> bool:
        text = str(exc)
        return (
            "Timeout: NTEvent" in text
            and '"result": 0' in text
            and '"errMsg": ""' in text
        )

    def _fact_check_forward_result(self, event: AstrMessageEvent, text: str):
        chunks = self._split_reply_text(text, max_chars=1200)
        nodes = []
        self_id = str(event.get_self_id() or "0")
        for index, chunk in enumerate(chunks, start=1):
            name = "事实核查" if len(chunks) == 1 else f"事实核查 {index}/{len(chunks)}"
            nodes.append(Node(uin=self_id, name=name, content=[Plain(chunk)]))
        return event.chain_result([Nodes(nodes)])

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
        yield event.plain_result(
            "用法：回复一条消息后发送 /事实核查，或者直接发送 /事实核查 要核查的内容。"
        )

    async def _build_fact_check_request(self, event: AstrMessageEvent) -> FactCheckRequest:
        trigger_text = event.message_str or ""
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
                local_texts: list[str] = []
                if comp.message_str:
                    local_texts.append(str(comp.message_str).strip())
                if comp.chain:
                    local_texts.extend(self._plain_texts(comp.chain))
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
            elif isinstance(comp, Image):
                images.extend(await self._image_inputs([comp], remaining=max_images - len(images)))

            if len(images) >= max_images:
                images = images[:max_images]

        text = "\n".join(part for part in quoted_texts if part).strip()
        if self._is_unusable_quoted_text(text):
            text = ""
        if inline_text:
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

    def _is_unusable_quoted_text(self, text: str | None) -> bool:
        if not isinstance(text, str):
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        placeholder_patterns = [
            r"^\[?(?:CQ:)?forward[,:\s].*id=\d+.*\]?$",
            r"^\[(?:合并转发|转发消息|forward message)[:：]?\d*\]$",
            r"^\[引用消息\]$",
            r"^\[Forward Message\]$",
            r"^\d{12,}$",
        ]
        return all(
            any(re.fullmatch(pattern, line, flags=re.IGNORECASE) for pattern in placeholder_patterns)
            for line in lines
        )

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
            url = str(comp.url or comp.file or "").strip()
            if not path and (not url or not url.startswith(("http://", "https://", "file://", "base64://"))):
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

    def _short_ref(self, value: str, limit: int = 120) -> str:
        value = str(value or "").replace("\n", " ").strip()
        return value if len(value) <= limit else value[:limit] + "..."
