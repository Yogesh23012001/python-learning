"""URL shortener service.

Provides a simple in-memory URL shortener with custom aliases,
expiration, and click tracking.
"""

from __future__ import annotations

import re
import secrets
import string
import time
from dataclasses import dataclass
from enum import StrEnum

# ============================================================
# Constants
# ============================================================

CODE_LENGTH = 7
CODE_ALPHABET = string.ascii_letters + string.digits
MAX_GENERATION_ATTEMPTS = 5
ALIAS_MIN_LENGTH = 1
ALIAS_MAX_LENGTH = 32
ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
RESERVED_ALIASES = frozenset({"admin", "api", "www", "login", "static"})


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


class InvalidAliasError(URLShortenerError):
    """Custom alias failed validation."""


# ============================================================
# Data model
# ============================================================


class URLStatus(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"


@dataclass
class ShortURL:
    code: str
    long_url: str
    created_at: float
    expires_at: float | None = None
    click_count: int = 0

    def is_expired(self, now: float) -> bool:
        """Return True if this URL has expired at the given time."""
        return self.expires_at is not None and now >= self.expires_at


# ============================================================
# Service
# ============================================================


class URLShortener:
    """In-memory URL shortener with TTL and custom aliases."""

    def __init__(self) -> None:
        self._storage: dict[str, ShortURL] = {}

    # --- Internal helpers --------------------------------

    def _now(self) -> float:
        """Return current unix time. Override in tests to control time."""
        return time.time()

    def _generate_code(self) -> str:
        """Generate a unique short code using cryptographic randomness.

        Raises:
            GenerationFailedError: If a unique code can't be generated
                after MAX_GENERATION_ATTEMPTS attempts.
        """
        for _ in range(MAX_GENERATION_ATTEMPTS):
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if code not in self._storage:
                return code
        raise GenerationFailedError(
            f"failed to generate unique code after {MAX_GENERATION_ATTEMPTS} attempts"
        )

    def _validate_alias(self, alias: str) -> None:
        """Validate a custom alias.

        Raises:
            InvalidAliasError: If the alias is empty, too long, contains
                disallowed characters, or is reserved.
        """
        if not (ALIAS_MIN_LENGTH <= len(alias) <= ALIAS_MAX_LENGTH):
            raise InvalidAliasError(
                f"alias must be {ALIAS_MIN_LENGTH}-{ALIAS_MAX_LENGTH} chars, got {len(alias)}"
            )
        if not ALIAS_PATTERN.match(alias):
            raise InvalidAliasError(
                f"alias '{alias}' contains invalid characters (allowed: A-Z, a-z, 0-9, _, -)"
            )
        if alias.lower() in RESERVED_ALIASES:
            raise InvalidAliasError(f"alias '{alias}' is reserved")

    # --- Public API --------------------------------------

    def shorten(
        self,
        long_url: str,
        *,
        custom_alias: str | None = None,
        ttl_seconds: float | None = None,
    ) -> ShortURL:
        """Create a short URL.

        Args:
            long_url: The URL to shorten. Must be non-empty after stripping.
            custom_alias: Optional custom code. If omitted, one is generated.
            ttl_seconds: Optional expiration in seconds from now.

        Returns:
            The newly created ShortURL record.

        Raises:
            ValueError: If long_url is empty or whitespace-only.
            InvalidAliasError: If custom_alias fails validation.
            CodeAlreadyExistsError: If custom_alias is already in use.
            GenerationFailedError: If auto-code generation fails.
        """
        long_url = long_url.strip()
        if not long_url:
            raise ValueError("long_url must be non-empty")

        if custom_alias is not None:
            self._validate_alias(custom_alias)
            if custom_alias in self._storage:
                raise CodeAlreadyExistsError(f"alias '{custom_alias}' is already in use")
            code = custom_alias
        else:
            code = self._generate_code()

        now = self._now()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        record = ShortURL(
            code=code,
            long_url=long_url,
            created_at=now,
            expires_at=expires_at,
        )
        self._storage[code] = record
        return record

    def resolve(self, code: str) -> str:
        """Resolve a short code to its long URL. Increments click count.

        Raises:
            CodeNotFoundError: If the code doesn't exist.
            CodeExpiredError: If the code has expired.
        """
        record = self._storage.get(code)
        if record is None:
            raise CodeNotFoundError(f"no URL found for code '{code}'")
        if record.is_expired(self._now()):
            raise CodeExpiredError(f"code '{code}' expired at {record.expires_at}")
        record.click_count += 1
        return record.long_url

    def top_urls(self, n: int) -> list[ShortURL]:
        """Return top-n URLs by click count, descending."""
        return sorted(
            self._storage.values(),
            key=lambda url: url.click_count,
            reverse=True,
        )[:n]
