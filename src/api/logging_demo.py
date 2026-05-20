"""Hour 1 — structlog demo."""

from __future__ import annotations

import contextlib

import structlog

from api.logging_config import configure_logging, get_logger

configure_logging(json_logs=True, level=20)  # 20 == logging.INFO; pretty output
logger = get_logger(__name__)


def divide(a: float, b: float) -> float:
    logger.info("dividing", a=a, b=b)
    try:
        return a / b
    except ZeroDivisionError:
        logger.exception("division failed", a=a, b=b)
        raise


def main() -> None:
    # Bind some context — every log within this scope includes user_id=42
    structlog.contextvars.bind_contextvars(user_id=42, request_id="req-abc")

    divide(10, 2)
    with contextlib.suppress(ZeroDivisionError):
        divide(10, 0)

    structlog.contextvars.clear_contextvars()


if __name__ == "__main__":
    main()
