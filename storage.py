from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp",
    )
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            temp_path.chmod(0o600)
        except OSError:
            pass
        temp_path.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_json_file(path: Path, default: Any = None) -> Any:
    target: Path | None = None
    try:
        target = Path(path)
        return json.loads(target.read_text(encoding="utf-8"))
    except OSError:
        return default
    except (ValueError, TypeError) as exc:
        if target is not None and target.exists():
            corrupt = target.with_name(f"{target.name}.corrupt-{time.time_ns()}")
            try:
                target.replace(corrupt)
                print(
                    f"[astrbot-fact-check-storage-corrupt] moved={corrupt.name} error={type(exc).__name__}",
                    flush=True,
                )
            except OSError:
                pass
        return default


class FactCheckMetricsStore:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._data = self._load()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "version": 2,
            "requests": 0,
            "success": 0,
            "partial": 0,
            "failure": 0,
            "cache_hits": 0,
            "followups": 0,
            "delivery_success": 0,
            "delivery_failure": 0,
            "elapsed_total": 0.0,
            "failure_stages": {},
        }

    def _load(self) -> dict[str, Any]:
        data = read_json_file(self.path, {}) if self.path is not None else {}
        loaded = self._empty()
        if not isinstance(data, dict):
            return loaded
        for key in (
            "requests",
            "success",
            "partial",
            "failure",
            "cache_hits",
            "followups",
            "delivery_success",
            "delivery_failure",
        ):
            loaded[key] = max(0, int(data.get(key) or 0))
        loaded["elapsed_total"] = max(0.0, float(data.get("elapsed_total") or 0.0))
        stages = data.get("failure_stages") or {}
        if isinstance(stages, dict):
            loaded["failure_stages"] = {
                str(key): max(0, int(value or 0))
                for key, value in stages.items()
                if str(key).strip()
            }
        return loaded

    def record(
        self,
        *,
        outcome: str,
        elapsed: float,
        cache_hit: bool = False,
        followup: bool = False,
        failure_stage: str = "",
    ) -> dict[str, Any]:
        normalized = (
            outcome if outcome in {"success", "partial", "failure"} else "failure"
        )
        with self._lock:
            self._data["requests"] += 1
            self._data[normalized] += 1
            self._data["elapsed_total"] += max(0.0, float(elapsed or 0.0))
            if cache_hit:
                self._data["cache_hits"] += 1
            if followup:
                self._data["followups"] += 1
            stage = str(failure_stage or "").strip()
            if normalized == "failure" and stage:
                stages = self._data["failure_stages"]
                stages[stage] = stages.get(stage, 0) + 1
            if self.path is not None:
                atomic_write_json(self.path, self._data)
            return self.snapshot()

    def record_delivery(self, *, success: bool) -> dict[str, Any]:
        with self._lock:
            key = "delivery_success" if success else "delivery_failure"
            self._data[key] += 1
            if self.path is not None:
                atomic_write_json(self.path, self._data)
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._data)
            result["failure_stages"] = dict(self._data["failure_stages"])
            requests = max(1, int(result["requests"]))
            result["average_seconds"] = round(
                float(result["elapsed_total"]) / requests, 2
            )
            return result

    def render_status(self) -> str:
        metrics = self.snapshot()
        stages = metrics["failure_stages"]
        stage_text = (
            "、".join(f"{key}={value}" for key, value in sorted(stages.items())) or "无"
        )
        return (
            "事实核查状态\n"
            f"累计请求：{metrics['requests']}\n"
            f"完整成功：{metrics['success']}\n"
            f"部分完成：{metrics['partial']}\n"
            f"失败：{metrics['failure']}\n"
            f"缓存命中：{metrics['cache_hits']}\n"
            f"追问：{metrics['followups']}\n"
            f"发送成功：{metrics['delivery_success']}\n"
            f"发送失败：{metrics['delivery_failure']}\n"
            f"平均耗时：{metrics['average_seconds']:.2f} 秒\n"
            f"失败阶段：{stage_text}"
        )
