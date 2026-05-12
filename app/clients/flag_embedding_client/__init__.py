from __future__ import annotations

from app.domain.embedding_runtime_vectors import coerce_colbert_vectors
from app.errors import EmbeddingModelError

from .batch_executor import EmbeddingBatchExecutor
from .client import FlagEmbeddingClient
from .factory import get_flag_embedding_client, reset_flag_embedding_client
from .models import EmbeddingTask
from .queue_manager import EmbeddingTaskQueue
from .runtime import FlagEmbeddingRuntime
from .settings import FlagEmbeddingClientSettings

__all__ = [
    "coerce_colbert_vectors",
    "EmbeddingBatchExecutor",
    "EmbeddingModelError",
    "EmbeddingTask",
    "EmbeddingTaskQueue",
    "FlagEmbeddingClient",
    "FlagEmbeddingClientSettings",
    "FlagEmbeddingRuntime",
    "get_flag_embedding_client",
    "reset_flag_embedding_client",
]
