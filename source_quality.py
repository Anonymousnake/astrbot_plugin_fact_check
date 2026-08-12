from __future__ import annotations

import re
from urllib.parse import urlparse


_GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_MULTI_LABEL_SUFFIXES = {
    "ac.uk",
    "co.jp",
    "co.uk",
    "com.au",
    "com.cn",
    "com.hk",
    "edu.cn",
    "gov.cn",
    "gov.uk",
    "net.cn",
    "org.cn",
    "org.uk",
}
_LOW_TRUST_HOSTS = {
    "blogspot.com",
    "facebook.com",
    "instagram.com",
    "medium.com",
    "reddit.com",
    "substack.com",
    "tiktok.com",
    "tumblr.com",
    "wordpress.com",
    "x.com",
}
_LOW_TRUST_TERMS = re.compile(
    r"(?:博客|论坛|个人主页|自媒体|社交平台|\bblog\b|\bforum\b|\bsocial\b)",
    flags=re.IGNORECASE,
)


def _source_url(source: str) -> str:
    match = re.search(r"https?://[^\s；;，,]+", str(source or ""))
    return match.group(0).rstrip(".。)]）") if match else ""


def _normalized_host(source: str) -> str:
    host = (urlparse(_source_url(source)).hostname or "").lower().rstrip(".")
    return host.removeprefix("www.")


def _host_matches(host: str, suffix: str) -> bool:
    normalized = str(suffix or "").lower().strip(".")
    return bool(host and normalized and (host == normalized or host.endswith("." + normalized)))


def _registered_domain(host: str) -> str:
    labels = [part for part in str(host or "").split(".") if part]
    if len(labels) <= 2:
        return ".".join(labels)
    suffix = ".".join(labels[-2:])
    if suffix in _MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix


def source_identity(source: str) -> str:
    """Return an organization-level identity suitable for independence checks."""
    host = _normalized_host(source)
    if not host:
        return ""
    if host == _GROUNDING_REDIRECT_HOST:
        # Redirect targets are opaque until resolved. Different labels or paths do
        # not prove that the underlying publishers are independent.
        return "grounding:unresolved"
    return _registered_domain(host)


def is_primary_source(source: str) -> bool:
    """Recognize official authorities without treating arbitrary schools as one."""
    host = _normalized_host(source)
    if not host:
        return False
    if _host_matches(host, "gov.uk") or _host_matches(host, "gov.cn"):
        return True
    if host.endswith(".gov") or host.endswith(".mil"):
        return True
    return any(
        _host_matches(host, authority)
        for authority in ("who.int", "un.org", "europa.eu")
    )


def _is_low_trust_source(source: str) -> bool:
    host = _normalized_host(source)
    if not host or host == _GROUNDING_REDIRECT_HOST:
        return True
    if any(_host_matches(host, blocked) for blocked in _LOW_TRUST_HOSTS):
        return True
    if host.startswith(("blog.", "blogs.")) or ".blog." in host:
        return True
    return bool(_LOW_TRUST_TERMS.search(str(source or "")))


def has_strong_claim_evidence(sources: list[str]) -> bool:
    """Require an authority or two credible, organization-independent sources."""
    clean_sources = [
        str(source or "").strip() for source in sources if str(source or "").strip()
    ]
    if any(is_primary_source(source) for source in clean_sources):
        return True
    identities: set[str] = set()
    for source in clean_sources:
        if _is_low_trust_source(source):
            continue
        identity = source_identity(source)
        if identity:
            identities.add(identity)
    return len(identities) >= 2
