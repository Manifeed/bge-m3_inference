from __future__ import annotations

import os


CANONICAL_BGE_M3_MODEL_NAME = "bge-m3"
BGE_M3_MODEL_ALIASES = frozenset({"bge-m3", "BAAI/bge-m3"})
DEFAULT_MODEL_PATH = "/opt/models/bge-m3"
DEFAULT_SERVER_PORT = 8000


class EmbeddingServiceConfigurationError(RuntimeError):
    """Raised when the embedding service is missing required configuration."""


def resolve_embedding_api_key() -> str:
    return os.getenv("EMBEDDING_SERVICE_API_KEY", "").strip()


def require_embedding_api_key() -> str:
    value = resolve_embedding_api_key()
    if not value:
        raise EmbeddingServiceConfigurationError("EMBEDDING_SERVICE_API_KEY is required")
    return value


def resolve_model_source() -> str:
    configured_path = os.getenv("SOURCE_EMBEDDING_MODEL_PATH", "").strip()
    return configured_path or DEFAULT_MODEL_PATH
