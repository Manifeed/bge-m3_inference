from __future__ import annotations

import logging
import time

from app.domain.batching import EmbeddingModeSignature
from app.errors import EmbeddingModelError
from app.schemas.embedding_schema import SparseEmbeddingRead

from .models import EmbeddingTask
from .queue_manager import EmbeddingTaskQueue
from .runtime import FlagEmbeddingRuntime


logger = logging.getLogger(__name__)


class EmbeddingBatchExecutor:
    def __init__(self, *, runtime: FlagEmbeddingRuntime, task_queue: EmbeddingTaskQueue) -> None:
        self._runtime = runtime
        self._task_queue = task_queue

    def process_batch(self, batch: list[EmbeddingTask]) -> None:
        signature = batch[0].signature
        texts = [task.normalized_text for task in batch]
        estimated_tokens = sum(task.token_estimate for task in batch)
        batch_size = self._task_queue.effective_batch_size(signature=signature, task_count=len(texts))
        start_time = time.perf_counter()

        dense_vectors, sparse_vectors, colbert_vectors = self._encode_with_retry(
            texts=texts,
            signature=signature,
            batch_size=batch_size,
        )

        self._log_completed_batch(
            batch=batch,
            estimated_tokens=estimated_tokens,
            duration_seconds=max(time.perf_counter() - start_time, 1e-9),
            batch_size=batch_size,
            signature=signature,
        )
        self._complete_tasks(
            batch=batch,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            colbert_vectors=colbert_vectors,
        )

    def fail_tasks(self, tasks: list[EmbeddingTask], error: Exception) -> None:
        for task in tasks:
            if not task.future.done():
                task.future.set_exception(error)

    def _encode_with_retry(
        self,
        *,
        texts: list[str],
        signature: EmbeddingModeSignature,
        batch_size: int,
    ) -> tuple[list[list[float] | None], list[SparseEmbeddingRead | None], list[list[list[float]] | None]]:
        try:
            return self._runtime.encode_batch(texts=texts, signature=signature, batch_size=batch_size)
        except EmbeddingModelError as exception:
            if not self._runtime.is_cuda_out_of_memory_error(exception) or batch_size <= 1:
                raise

        reduced_batch_size = self._task_queue.reduce_batch_size(
            signature=signature,
            previous_batch_size=batch_size,
        )
        logger.warning(
            "bge-m3_inference OOM detected retrying dense=%s sparse=%s colbert=%s previous_batch_size=%s reduced_batch_size=%s",
            signature.dense,
            signature.sparse,
            signature.colbert,
            batch_size,
            reduced_batch_size,
        )
        return self._runtime.encode_batch(
            texts=texts,
            signature=signature,
            batch_size=reduced_batch_size,
        )

    def _log_completed_batch(
        self,
        *,
        batch: list[EmbeddingTask],
        estimated_tokens: int,
        duration_seconds: float,
        batch_size: int,
        signature: EmbeddingModeSignature,
    ) -> None:
        tokens_per_second = estimated_tokens / duration_seconds
        logger.info(
            "bge-m3_inference batch completed items=%s estimated_tokens=%s duration_s=%.4f tokens_per_second=%.2f effective_batch_size=%s dense=%s sparse=%s colbert=%s",
            len(batch),
            estimated_tokens,
            duration_seconds,
            tokens_per_second,
            batch_size,
            signature.dense,
            signature.sparse,
            signature.colbert,
        )

    def _complete_tasks(
        self,
        *,
        batch: list[EmbeddingTask],
        dense_vectors: list[list[float] | None],
        sparse_vectors: list[SparseEmbeddingRead | None],
        colbert_vectors: list[list[list[float]] | None],
    ) -> None:
        for index, task in enumerate(batch):
            if not task.future.done():
                task.future.set_result(
                    (
                        dense_vectors[index],
                        sparse_vectors[index],
                        colbert_vectors[index],
                    )
                )
