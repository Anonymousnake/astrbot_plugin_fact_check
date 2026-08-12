from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .fact_check import FactCheckRequest


def build_fact_check_kwargs(
    config: Any,
    request_data: FactCheckRequest,
    timeout_seconds: float,
    *,
    list_config: Callable[[str, list[str]], list[str]],
) -> dict[str, Any]:
    """Translate AstrBot config into the synchronous pipeline contract."""
    return {
        "request_data": request_data,
        "api_key": str(config.get("gemini_api_key") or ""),
        "base_url": str(
            config.get(
                "gemini_base_url",
                "https://generativelanguage.googleapis.com/v1beta/models",
            )
            or "https://generativelanguage.googleapis.com/v1beta/models"
        ),
        "pre_model": str(config.get("fact_check_pre_model") or "gemini-3.1-flash-lite"),
        "evidence_model": str(
            config.get("fact_check_evidence_model") or "gemini-2.5-flash"
        ).strip(),
        "verdict_models": list_config(
            "fact_check_verdict_models", ["gemini-3-flash-preview"]
        ),
        "max_image_bytes": int(
            config.get("fact_check_max_image_bytes") or 5 * 1024 * 1024
        ),
        "image_download_hard_limit_bytes": int(
            config.get("fact_check_image_download_hard_limit_bytes") or 20 * 1024 * 1024
        ),
        "image_max_pixels": int(
            config.get("fact_check_image_max_pixels") or 20_000_000
        ),
        "image_total_inline_bytes": int(
            config.get("fact_check_image_total_inline_bytes") or 10 * 1024 * 1024
        ),
        "long_image_chunk_height": int(
            config.get("fact_check_long_image_chunk_height") or 2200
        ),
        "long_image_max_parts": int(config.get("fact_check_long_image_max_parts") or 8),
        "long_image_max_width": int(
            config.get("fact_check_long_image_max_width") or 1280
        ),
        "image_download_timeout": int(
            config.get("fact_check_image_download_timeout_seconds") or 10
        ),
        "pre_request_timeout": int(config.get("fact_check_pre_timeout_seconds") or 25),
        "main_request_timeout": int(
            config.get("fact_check_main_timeout_seconds") or 45
        ),
        "evidence_max_output_tokens": int(
            config.get("fact_check_evidence_max_output_tokens") or 1536
        ),
        "evidence_retry_max_output_tokens": int(
            config.get("fact_check_evidence_retry_max_output_tokens") or 3072
        ),
        "anysearch_enabled": bool(config.get("fact_check_anysearch_enabled", False)),
        "anysearch_endpoint": str(
            config.get("fact_check_anysearch_endpoint")
            or "https://api.anysearch.com/mcp"
        ),
        "anysearch_api_key": str(config.get("fact_check_anysearch_api_key") or ""),
        "anysearch_timeout": int(
            config.get("fact_check_anysearch_timeout_seconds") or 20
        ),
        "anysearch_max_claims": int(config.get("fact_check_anysearch_max_claims") or 3),
        "anysearch_max_results_per_claim": int(
            config.get("fact_check_anysearch_max_results_per_claim") or 3
        ),
        "anysearch_extract_top_urls": int(
            config.get("fact_check_anysearch_extract_top_urls") or 2
        ),
        "anysearch_max_chars": int(
            config.get("fact_check_anysearch_max_chars") or 6000
        ),
        "anysearch_freshness": str(config.get("fact_check_anysearch_freshness") or ""),
        "anysearch_content_types": list_config(
            "fact_check_anysearch_content_types",
            ["web", "news"],
        ),
        "model_failure_cooldown_seconds": int(
            config.get("fact_check_model_failure_cooldown_seconds") or 900
        ),
        "verdict_request_timeout": int(
            config.get("fact_check_verdict_timeout_seconds") or 25
        ),
        "verdict_max_output_tokens": int(
            config.get("fact_check_verdict_max_output_tokens") or 2048
        ),
        "verdict_retry_max_output_tokens": int(
            config.get("fact_check_verdict_retry_max_output_tokens") or 4096
        ),
        "verdict_policy": str(config.get("fact_check_verdict_policy") or "risk_based"),
        "verdict_thinking_level": str(
            config.get("fact_check_verdict_thinking_level") or "medium"
        ),
        "total_timeout_seconds": int(timeout_seconds),
    }
