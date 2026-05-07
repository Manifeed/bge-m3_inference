from __future__ import annotations

import os
from threading import Lock
from typing import Any

from app.domain.config import CANONICAL_BGE_M3_MODEL_NAME, DEFAULT_MODEL_PATH
from app.schemas.embedding_schema import SparseEmbeddingRead


MODEL_PATH_ENV = "SOURCE_EMBEDDING_MODEL_PATH"
_MODEL: FlagEmbeddingClient | None = None
_MODEL_LOCK = Lock()


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding runtime cannot serve vectors."""


class FlagEmbeddingClient:
    def __init__(self, model_name: str = CANONICAL_BGE_M3_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model_source = _resolve_model_source()
        self._model = self._load_model(self.model_source)

    def encode(
        self,
        *,
        texts: list[str],
        dense: bool,
        sparse: bool,
    ) -> tuple[list[list[float]], list[SparseEmbeddingRead]]:
        normalized_texts = [_normalize_text(text) for text in texts]
        if not normalized_texts:
            return [], []

        try:
            output = self._model.encode(
                normalized_texts,
                return_dense=dense,
                return_sparse=sparse,
                return_colbert_vecs=False,
            )
        except Exception as exception:
            raise EmbeddingModelError("Unable to encode embeddings") from exception

        dense_vectors = _coerce_dense_vectors(output.get("dense_vecs"), count=len(texts), enabled=dense)
        sparse_vectors = _coerce_sparse_vectors(
            output.get("lexical_weights"),
            count=len(texts),
            enabled=sparse,
        )
        if len(dense_vectors) != len(texts) or len(sparse_vectors) != len(texts):
            raise EmbeddingModelError(
                f"Embedding runtime returned {len(dense_vectors)} dense and "
                f"{len(sparse_vectors)} sparse vectors for {len(texts)} inputs"
            )
        return dense_vectors, sparse_vectors

    def check_ready(self) -> None:
        self.encode(texts=["ready"], dense=True, sparse=True)

    def _load_model(self, model_source: str) -> Any:
        try:
            from FlagEmbedding import BGEM3FlagModel
            import torch
        except ImportError as exception:
            if isinstance(exception, ModuleNotFoundError) and exception.name == "FlagEmbedding":
                raise EmbeddingModelError("FlagEmbedding is not installed") from exception
            raise EmbeddingModelError(
                f"Unable to import FlagEmbedding runtime dependencies: {exception}"
            ) from exception

        if not torch.cuda.is_available():
            raise EmbeddingModelError("CUDA GPU is required but no GPU is available")

        try:
            return BGEM3FlagModel(model_source, use_fp16=True)
        except Exception as exception:
            raise EmbeddingModelError(f"Unable to load embedding model {self.model_name}") from exception


def get_flag_embedding_client() -> FlagEmbeddingClient:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = FlagEmbeddingClient()
    return _MODEL


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _resolve_model_source() -> str:
    configured_path = os.getenv(MODEL_PATH_ENV, "").strip()
    return configured_path or DEFAULT_MODEL_PATH


def _coerce_dense_vectors(value: Any, *, count: int, enabled: bool) -> list[list[float]]:
    if not enabled:
        return [[] for _ in range(count)]
    if value is None:
        return []
    vectors = value.tolist() if hasattr(value, "tolist") else value
    return [[float(item) for item in vector] for vector in vectors]


def _coerce_sparse_vectors(value: Any, *, count: int, enabled: bool) -> list[SparseEmbeddingRead]:
    if not enabled:
        return [SparseEmbeddingRead() for _ in range(count)]
    if value is None:
        return []
    items = value.tolist() if hasattr(value, "tolist") else value
    return [_coerce_sparse_vector(item) for item in items]


def _coerce_sparse_vector(value: Any) -> SparseEmbeddingRead:
    if isinstance(value, SparseEmbeddingRead):
        return value
    if not isinstance(value, dict):
        raise EmbeddingModelError("Sparse embedding payload must be a dictionary")

    pairs: list[tuple[int, float]] = []
    for raw_index, raw_weight in value.items():
        try:
            index = int(raw_index)
            weight = float(raw_weight)
        except (TypeError, ValueError) as exception:
            raise EmbeddingModelError("Sparse embedding payload contains invalid values") from exception
        if weight != 0.0:
            pairs.append((index, weight))
    pairs.sort(key=lambda item: item[0])
    return SparseEmbeddingRead(
        indices=[index for index, _ in pairs],
        values=[weight for _, weight in pairs],
    )
