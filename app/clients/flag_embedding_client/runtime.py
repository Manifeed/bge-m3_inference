from __future__ import annotations

from typing import Any

from app.domain.batching import EmbeddingModeSignature
from app.domain.embedding_runtime_vectors import (
    coerce_colbert_vectors,
    coerce_dense_vectors,
    coerce_sparse_vectors,
    is_cuda_out_of_memory_error,
)
from app.errors import EmbeddingModelError
from app.schemas.embedding_schema import SparseEmbeddingRead


class FlagEmbeddingRuntime:
    def __init__(
        self,
        *,
        model_name: str,
        model_source: str,
        max_length: int,
        use_fp16: bool,
    ) -> None:
        self.model_name = model_name
        self.model_source = model_source
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel
            import torch
            from transformers import AutoTokenizer
        except ImportError as exception:
            if isinstance(exception, ModuleNotFoundError) and exception.name == "FlagEmbedding":
                raise EmbeddingModelError("FlagEmbedding is not installed") from exception
            raise EmbeddingModelError(
                f"Unable to import FlagEmbedding runtime dependencies: {exception}"
            ) from exception

        if not torch.cuda.is_available():
            raise EmbeddingModelError("CUDA GPU is required but no GPU is available")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_source, local_files_only=True)
            self._model = BGEM3FlagModel(self.model_source, use_fp16=self.use_fp16)
        except Exception as exception:
            raise EmbeddingModelError(f"Unable to load embedding model {self.model_name}") from exception

    def encode_batch(
        self,
        *,
        texts: list[str],
        signature: EmbeddingModeSignature,
        batch_size: int,
    ) -> tuple[list[list[float] | None], list[SparseEmbeddingRead | None], list[list[list[float]] | None]]:
        if self._model is None:
            raise EmbeddingModelError("Embedding model is not loaded")
        try:
            output = self._model.encode(
                texts,
                batch_size=batch_size,
                max_length=self.max_length,
                return_dense=signature.dense,
                return_sparse=signature.sparse,
                return_colbert_vecs=signature.colbert,
            )
        except Exception as exception:
            raise EmbeddingModelError(f"Unable to encode embeddings: {exception}") from exception

        dense_vectors = coerce_dense_vectors(output.get("dense_vecs"), count=len(texts), enabled=signature.dense)
        sparse_vectors = coerce_sparse_vectors(
            output.get("lexical_weights"),
            count=len(texts),
            enabled=signature.sparse,
        )
        colbert_vectors = coerce_colbert_vectors(
            output.get("colbert_vecs"),
            count=len(texts),
            enabled=signature.colbert,
        )
        if (
            len(dense_vectors) != len(texts)
            or len(sparse_vectors) != len(texts)
            or len(colbert_vectors) != len(texts)
        ):
            raise EmbeddingModelError(
                f"Embedding runtime returned {len(dense_vectors)} dense, "
                f"{len(sparse_vectors)} sparse, and {len(colbert_vectors)} ColBERT vectors "
                f"for {len(texts)} inputs"
            )
        return dense_vectors, sparse_vectors, colbert_vectors

    def estimate_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return min(max(len(text.split()), 1), self.max_length)
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        return max(1, len(token_ids))

    def is_cuda_out_of_memory_error(self, exception: EmbeddingModelError) -> bool:
        return is_cuda_out_of_memory_error(exception)
