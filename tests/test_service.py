"""Tests for the URL shortener service."""

from __future__ import annotations

import pytest
from url_shortener.service import (
    CodeAlreadyExistsError,
    CodeExpiredError,
    CodeNotFoundError,
    InvalidAliasError,
    URLShortener,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def svc() -> URLShortener:
    """Fresh URLShortener instance per test."""
    return URLShortener()


class FakeClock:
    """Controllable clock for time-dependent tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


@pytest.fixture
def svc_with_clock() -> tuple[URLShortener, FakeClock]:
    """Service with overridable clock."""
    s = URLShortener()
    clock = FakeClock()
    s._now = clock  # type: ignore[method-assign]
    return s, clock


# ============================================================
# Tests: shorten()
# ============================================================


def test_shorten_returns_record_with_auto_code(svc: URLShortener) -> None:
    url = svc.shorten("https://example.com")
    assert url.long_url == "https://example.com"
    assert len(url.code) == 7
    assert url.click_count == 0
    assert url.expires_at is None


def test_shorten_strips_whitespace(svc: URLShortener) -> None:
    url = svc.shorten("  https://example.com  ")
    assert url.long_url == "https://example.com"


def test_shorten_raises_on_empty_url(svc: URLShortener) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        svc.shorten("")


def test_shorten_raises_on_whitespace_url(svc: URLShortener) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        svc.shorten("   ")


def test_shorten_with_custom_alias(svc: URLShortener) -> None:
    url = svc.shorten("https://google.com", custom_alias="google")
    assert url.code == "google"


def test_shorten_raises_on_duplicate_alias(svc: URLShortener) -> None:
    svc.shorten("https://google.com", custom_alias="google")
    with pytest.raises(CodeAlreadyExistsError, match="google"):
        svc.shorten("https://other.com", custom_alias="google")


@pytest.mark.parametrize(
    "bad_alias",
    [
        "",  # empty
        "a" * 33,  # too long
        "has space",  # space disallowed
        "has/slash",  # slash disallowed
        "admin",  # reserved
        "API",  # reserved (case-insensitive)
    ],
)
def test_shorten_rejects_invalid_aliases(svc: URLShortener, bad_alias: str) -> None:
    with pytest.raises(InvalidAliasError):
        svc.shorten("https://example.com", custom_alias=bad_alias)


# ============================================================
# Tests: resolve()
# ============================================================


def test_resolve_returns_long_url(svc: URLShortener) -> None:
    svc.shorten("https://example.com", custom_alias="ex")
    assert svc.resolve("ex") == "https://example.com"


def test_resolve_increments_click_count(svc: URLShortener) -> None:
    url = svc.shorten("https://example.com", custom_alias="ex")
    svc.resolve("ex")
    svc.resolve("ex")
    svc.resolve("ex")
    assert url.click_count == 3


def test_resolve_raises_on_missing_code(svc: URLShortener) -> None:
    with pytest.raises(CodeNotFoundError, match="nonexistent"):
        svc.resolve("nonexistent")


def test_resolve_raises_on_expired_code(
    svc_with_clock: tuple[URLShortener, FakeClock],
) -> None:
    s, clock = svc_with_clock
    s.shorten("https://example.com", custom_alias="temp", ttl_seconds=10.0)
    clock.advance(11.0)  # advance past expiry
    with pytest.raises(CodeExpiredError, match="temp"):
        s.resolve("temp")


def test_resolve_works_before_expiry(
    svc_with_clock: tuple[URLShortener, FakeClock],
) -> None:
    s, clock = svc_with_clock
    s.shorten("https://example.com", custom_alias="temp", ttl_seconds=10.0)
    clock.advance(5.0)  # still within TTL
    assert s.resolve("temp") == "https://example.com"


# ============================================================
# Tests: top_urls()
# ============================================================


def test_top_urls_returns_by_click_count_desc(svc: URLShortener) -> None:
    svc.shorten("https://a.com", custom_alias="a")
    svc.shorten("https://b.com", custom_alias="b")
    svc.shorten("https://c.com", custom_alias="c")

    svc.resolve("b")
    svc.resolve("b")
    svc.resolve("b")
    svc.resolve("c")
    svc.resolve("c")
    svc.resolve("a")

    top = svc.top_urls(n=3)
    assert [u.code for u in top] == ["b", "c", "a"]


def test_top_urls_respects_n(svc: URLShortener) -> None:
    for i in range(5):
        svc.shorten(f"https://example.com/{i}", custom_alias=f"a{i}")
    assert len(svc.top_urls(n=2)) == 2


def test_top_urls_empty_storage(svc: URLShortener) -> None:
    assert svc.top_urls(n=5) == []
