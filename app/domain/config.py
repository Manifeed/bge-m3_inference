from __future__ import annotations

import os

from app.errors import EmbeddingServiceConfigurationError


CANONICAL_BGE_M3_MODEL_NAME = "bge-m3"
BGE_M3_MODEL_ALIASES = frozenset({"bge-m3", "BAAI/bge-m3"})
CANONICAL_SERVICE_NAME = "bge-m3_inference"
DEFAULT_MODEL_PATH = "/opt/models/bge-m3"
DEFAULT_SERVER_PORT = 8000
DEFAULT_BATCH_MAX_ITEMS = 64
DEFAULT_BATCH_MAX_TOKENS = 16_384
DEFAULT_BATCH_MAX_WAIT_MS = 5
DEFAULT_QUEUE_MAX_ITEMS = 1024
DEFAULT_MAX_LENGTH = 512
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_SHUTDOWN_GRACE_SECONDS = 30.0
DEFAULT_USE_FP16 = True


def resolve_embedding_api_key() -> str:
    return _first_non_empty_env("BGE_M3_INFERENCE_API_KEY", "EMBEDDING_SERVICE_API_KEY")


def require_embedding_api_key() -> str:
    value = resolve_embedding_api_key()
    if not value:
        raise EmbeddingServiceConfigurationError(
            "BGE_M3_INFERENCE_API_KEY is required "
            "(legacy EMBEDDING_SERVICE_API_KEY is still accepted temporarily)"
        )
    return value


def resolve_model_source() -> str:
    configured_path = os.getenv("SOURCE_EMBEDDING_MODEL_PATH", "").strip()
    return configured_path or DEFAULT_MODEL_PATH


def resolve_batch_max_items() -> int:
    return _positive_int_env("BGE_M3_INFERENCE_BATCH_MAX_ITEMS", default=DEFAULT_BATCH_MAX_ITEMS)


def resolve_batch_max_tokens() -> int:
    return _positive_int_env("BGE_M3_INFERENCE_BATCH_MAX_TOKENS", default=DEFAULT_BATCH_MAX_TOKENS)


def resolve_batch_max_wait_ms() -> int:
    return _positive_int_env("BGE_M3_INFERENCE_BATCH_MAX_WAIT_MS", default=DEFAULT_BATCH_MAX_WAIT_MS)


def resolve_queue_max_items() -> int:
    return _positive_int_env("BGE_M3_INFERENCE_QUEUE_MAX_ITEMS", default=DEFAULT_QUEUE_MAX_ITEMS)


def resolve_max_length() -> int:
    return _positive_int_env("BGE_M3_INFERENCE_MAX_LENGTH", default=DEFAULT_MAX_LENGTH)


def resolve_request_timeout_seconds() -> float:
    return _positive_float_env(
        "BGE_M3_INFERENCE_REQUEST_TIMEOUT_SECONDS",
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )


def resolve_shutdown_grace_seconds() -> float:
    return _positive_float_env(
        "BGE_M3_INFERENCE_SHUTDOWN_GRACE_SECONDS",
        default=DEFAULT_SHUTDOWN_GRACE_SECONDS,
    )


def resolve_use_fp16() -> bool:
    return _bool_env("BGE_M3_INFERENCE_USE_FP16", default=DEFAULT_USE_FP16)


def _first_non_empty_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _positive_int_env(name: str, *, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def _positive_float_env(name: str, *, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = float(raw_value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def _bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default
