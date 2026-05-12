from __future__ import annotations

import threading

from .client import FlagEmbeddingClient


_MODEL: FlagEmbeddingClient | None = None
_MODEL_LOCK = threading.Lock()


def get_flag_embedding_client() -> FlagEmbeddingClient:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = FlagEmbeddingClient()
    return _MODEL


def reset_flag_embedding_client() -> None:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            _MODEL.stop()
        _MODEL = None
