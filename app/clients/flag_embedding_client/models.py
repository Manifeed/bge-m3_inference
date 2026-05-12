from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass

from app.domain.batching import EmbeddingModeSignature
from app.schemas.embedding_schema import SparseEmbeddingRead


EmbeddingResult = tuple[list[float] | None, SparseEmbeddingRead | None, list[list[float]] | None]


@dataclass
class EmbeddingTask:
    normalized_text: str
    signature: EmbeddingModeSignature
    token_estimate: int
    future: Future[EmbeddingResult]
    enqueued_at: float
