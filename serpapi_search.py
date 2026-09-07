from __future__ import annotations

import http.client
import json
import time
from typing import TypedDict
from urllib.parse import urlencode, urlparse


class SearchHit(TypedDict):
    title: str
    url: str
    snippet: str


def search(
    query: str,
    *,
    api_key: str,
    timeout: float,
    max_results: int = 3,
    freshness: str = "",
) -> list[SearchHit]:
    """Query SerpAPI without exposing its query-string credential in HTTP logs.

    Args:
        query: Claim to search for.
        api_key: SerpAPI credential.
        timeout: Remaining request budget in seconds.
        max_results: Maximum organic results to return.
        freshness: Optional day, week, month, or year restriction.

    Returns:
        Bounded organic results; snippets remain unverified search hints.

    Raises:
        RuntimeError: The provider failed or returned malformed data.
    """
    deadline = time.monotonic() + max(0.01, float(timeout))
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": "zh-cn",
        "num": max(1, min(10, max_results)),
    }
    period = {"day": "d", "week": "w", "month": "m", "year": "y"}.get(freshness)
    if period:
        params["tbs"] = f"qdr:{period}"
    connection = http.client.HTTPSConnection(
        "serpapi.com", timeout=min(8.0, max(0.01, float(timeout)))
    )
    connection.set_debuglevel(0)
    try:
        connection.request(
            "GET",
            "/search.json?" + urlencode(params),
            headers={"Accept": "application/json"},
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("SerpAPI deadline exceeded")
        if connection.sock is not None:
            connection.sock.settimeout(min(8.0, remaining))
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"SerpAPI HTTP {response.status}")
        raw = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SerpAPI deadline exceeded")
            if connection.sock is not None:
                connection.sock.settimeout(min(8.0, remaining))
            chunk = response.read1(min(65536, 1_000_001 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > 1_000_000:
                raise RuntimeError("SerpAPI response exceeds size limit")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("error"):
            raise RuntimeError("SerpAPI returned a search error")
        rows = data.get("organic_results") or []
        if not isinstance(rows, list):
            raise TypeError("SerpAPI returned malformed organic results")
        hits: list[SearchHit] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("link"), str):
                continue
            url = row["link"].strip()
            try:
                parsed = urlparse(url)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or len(url) > 2048
                ):
                    continue
            except ValueError:
                continue
            hits.append(
                {
                    "title": str(row.get("title") or "")[:200],
                    "url": url,
                    "snippet": str(row.get("snippet") or "")[:600],
                }
            )
            if len(hits) >= max(1, min(10, max_results)):
                break
        return hits
    except (OSError, http.client.HTTPException, ValueError, TypeError) as exc:
        raise RuntimeError(f"SerpAPI request failed: {type(exc).__name__}") from None
    finally:
        connection.close()
