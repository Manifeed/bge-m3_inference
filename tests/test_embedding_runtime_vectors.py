from __future__ import annotations

import pytest

from app.clients.flag_embedding_client import EmbeddingModelError, coerce_colbert_vectors
from app.domain.embedding_runtime_vectors import (
    coerce_dense_vectors,
    coerce_sparse_vector,
    normalize_embedding_text,
)
from app.schemas.embedding_schema import SparseEmbeddingRead


def test_coerce_colbert_vectors_returns_none_when_disabled() -> None:
    assert coerce_colbert_vectors(None, count=2, enabled=False) == [None, None]


def test_coerce_colbert_vectors_converts_nested_values() -> None:
    vectors = coerce_colbert_vectors(
        [[[1, 2], [3.5, 4]], [[5, 6]]],
        count=2,
        enabled=True,
    )

    assert vectors == [
        [[1.0, 2.0], [3.5, 4.0]],
        [[5.0, 6.0]],
    ]


def test_coerce_colbert_vectors_rejects_missing_payload() -> None:
    with pytest.raises(EmbeddingModelError, match="ColBERT"):
        coerce_colbert_vectors(None, count=1, enabled=True)


def test_coerce_dense_vectors_returns_none_when_disabled() -> None:
    assert coerce_dense_vectors(None, count=2, enabled=False) == [None, None]


def test_coerce_sparse_vector_sorts_and_filters_zero_weights() -> None:
    vector = coerce_sparse_vector({"7": 0, "2": "1.5", "4": -0.25})

    assert vector == SparseEmbeddingRead(indices=[2, 4], values=[1.5, -0.25])


def test_normalize_embedding_text_only_trims_edges() -> None:
    assert normalize_embedding_text("  hello   world\nnext line  ") == "hello   world\nnext line"
