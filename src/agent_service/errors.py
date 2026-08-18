from __future__ import annotations

from enum import StrEnum

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import ValidationError


class FailureCategory(StrEnum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PROVIDER = "provider"
    SCHEMA = "schema"
    POLICY = "policy"
    UNKNOWN = "unknown"


def classify_exception(exc: BaseException) -> FailureCategory:
    if isinstance(exc, RateLimitError):
        return FailureCategory.RATE_LIMIT
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return FailureCategory.TIMEOUT
    if isinstance(exc, APIConnectionError):
        return FailureCategory.NETWORK
    if isinstance(exc, ValidationError):
        return FailureCategory.SCHEMA
    if isinstance(exc, InternalServerError):
        return FailureCategory.PROVIDER
    return FailureCategory.UNKNOWN
