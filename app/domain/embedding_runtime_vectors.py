from __future__ import annotations

from typing import Any

from app.errors import EmbeddingModelError
from app.schemas.embedding_schema import SparseEmbeddingRead


def normalize_embedding_text(value: str) -> str:
    return value.strip()


def coerce_dense_vectors(value: Any, *, count: int, enabled: bool) -> list[list[float] | None]:
    if not enabled:
        return [None for _ in range(count)]
    if value is None:
        return []
    vectors = value.tolist() if hasattr(value, "tolist") else value
    return [[float(item) for item in vector] for vector in vectors]


def coerce_sparse_vectors(value: Any, *, count: int, enabled: bool) -> list[SparseEmbeddingRead | None]:
    if not enabled:
        return [None for _ in range(count)]
    if value is None:
        return []
    items = value.tolist() if hasattr(value, "tolist") else value
    return [coerce_sparse_vector(item) for item in items]


def coerce_colbert_vectors(value: Any, *, count: int, enabled: bool) -> list[list[list[float]] | None]:
    if not enabled:
        return [None for _ in range(count)]
    if value is None:
        raise EmbeddingModelError("ColBERT embedding payload is missing")
    items = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(items, list):
        raise EmbeddingModelError("ColBERT embedding payload must be a list")
    vectors = [coerce_colbert_vector(item) for item in items]
    if any(len(vector) == 0 for vector in vectors):
        raise EmbeddingModelError("ColBERT embedding payload contains an empty token matrix")
    return vectors


def coerce_sparse_vector(value: Any) -> SparseEmbeddingRead:
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


def coerce_colbert_vector(value: Any) -> list[list[float]]:
    rows = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(rows, list):
        raise EmbeddingModelError("ColBERT embedding payload must contain a token matrix")
    matrix: list[list[float]] = []
    for row in rows:
        values = row.tolist() if hasattr(row, "tolist") else row
        if not isinstance(values, list):
            raise EmbeddingModelError("ColBERT embedding rows must be lists")
        matrix.append([float(item) for item in values])
    return matrix


def is_cuda_out_of_memory_error(exception: EmbeddingModelError) -> bool:
    return "out of memory" in str(exception).lower()
