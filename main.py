"""URL shortener — Hour 4 build."""

from __future__ import annotations

import random
import string
import time
import secrets
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Constants
# ============================================================

CODE_LENGTH = 7
CODE_ALPHABET = string.ascii_letters + string.digits  # 62 chars
MAX_GENERATION_ATTEMPTS = 5


# ============================================================
# Exceptions
# ============================================================

class URLShortenerError(Exception):
    """Base exception for all shortener errors."""


class CodeAlreadyExistsError(URLShortenerError):
    """Custom alias collides with an existing code."""


class CodeNotFoundError(URLShortenerError):
    """No URL exists for the given code."""


class CodeExpiredError(URLShortenerError):
    """The URL exists but has expired."""


class GenerationFailedError(URLShortenerError):
    """Could not generate a unique code after multiple attempts."""


# ============================================================
# Data model
# ============================================================

class URLStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"


@dataclass
class ShortURL:
    code: str
    long_url: str
    created_at: float                       # unix timestamp
    expires_at: float | None = None         # None = never expires
    click_count: int = 0

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    @property
    def status(self) -> URLStatus:
        return URLStatus.EXPIRED if self.is_expired(time.time()) else URLStatus.ACTIVE


# ============================================================
# Service
# ============================================================

class URLShortener:
    def __init__(self) -> None:
        self._storage: dict[str, ShortURL] = {}

    # --- Internal helpers --------------------------------

    def _generate_code(self) -> str:
        """Generate a random unique code; retry on collision."""
        for _ in range(MAX_GENERATION_ATTEMPTS):
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if code not in self._storage:
                return code
        raise GenerationFailedError

    def _now(self) -> float:
        """Wrapped for testability — override in tests if needed."""
        return time.time()

    # --- Public API --------------------------------------

    def shorten(
        self,
        long_url: str,
        *,
        custom_alias: str | None = None,
        ttl_seconds: float | None = None,
    ) -> ShortURL:
        """Create a short URL. Returns the ShortURL record."""
        long_url = long_url.strip()
        if not long_url:
            raise ValueError("long_url must be non-empty")
        
        if custom_alias is not None:    
            if custom_alias in self._storage:
                raise CodeAlreadyExistsError
            code = custom_alias
        else:
            code = self._generate_code()
        now = self._now()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        short_url = ShortURL(code=code, long_url=long_url, created_at=now, expires_at=expires_at)
        self._storage[code] = short_url   
        return short_url 

    def resolve(self, code: str) -> str:
        """Resolve a short code to its long URL. Increments click count.
        Raises CodeNotFoundError if missing, CodeExpiredError if expired.
        """
        # TODO: YOU IMPLEMENT THIS
        short_url = self._storage.get(code)
        if short_url is None:
            raise CodeNotFoundError
        now = self._now()

        if short_url.is_expired(now):
            raise CodeExpiredError
        short_url.click_count += 1
        return short_url.long_url   


    def top_urls(self, n: int) -> list[ShortURL]:
        """Return top-n URLs by click count, descending."""
        sorted_urls = sorted(self._storage.values(), key=lambda url: url.click_count, reverse=True)
        return sorted_urls[:n]
    

def main() -> None:
    svc = URLShortener()

    # Test 1: shorten + resolve
    url1 = svc.shorten("https://anthropic.com")
    print(f"Created code: {url1.code} for {url1.long_url}")
    resolved = svc.resolve(url1.code)
    print(f"Resolved: {resolved}")
    print(f"Click count after 1 resolve: {url1.click_count}")

    # Test 2: custom alias
    url2 = svc.shorten("https://google.com", custom_alias="google")
    print(f"Custom alias works: {url2.code}")
    print(f"Resolve 'google': {svc.resolve('google')}")

    # Test 3: alias collision
    try:
        svc.shorten("https://different.com", custom_alias="google")
    except CodeAlreadyExistsError as e:
        print(f"Got expected collision error")

    # Test 4: not found
    try:
        svc.resolve("nonexistent")
    except CodeNotFoundError:
        print(f"Got expected not-found error")

    # Test 5: expiration
    url3 = svc.shorten("https://temporary.com", ttl_seconds=0.1)
    print(f"Before expiry, resolve works: {svc.resolve(url3.code)}")
    time.sleep(0.2)
    try:
        svc.resolve(url3.code)
    except CodeExpiredError:
        print(f"Got expected expired error")

    # Test 6: top URLs by clicks
    svc.resolve("google")
    svc.resolve("google")
    svc.resolve("google")
    top = svc.top_urls(n=2)
    print(f"Top URLs:")
    for u in top:
        print(f"  {u.code}: {u.click_count} clicks ({u.long_url})")

    # Test 7: empty URL validation
    try:
        svc.shorten("   ")
    except ValueError:
        print(f"Got expected empty-URL error")


if __name__ == "__main__":
    main()
