from __future__ import annotations

import logging
import threading
import time

from app.domain.batching import (
    EmbeddingModeSignature,
    effective_max_items,
    effective_max_tokens,
    select_batch_indexes,
    selected_token_count,
)

from .models import EmbeddingTask
from .settings import FlagEmbeddingClientSettings


logger = logging.getLogger(__name__)


class EmbeddingTaskQueue:
    def __init__(self, *, condition: threading.Condition, settings: FlagEmbeddingClientSettings) -> None:
        self._condition = condition
        self._settings = settings
        self._pending: list[EmbeddingTask] = []
        self._adaptive_batch_items: dict[EmbeddingModeSignature, int] = {}

    def size(self) -> int:
        return len(self._pending)

    def is_full(self) -> bool:
        return len(self._pending) >= self._settings.queue_max_items

    def enqueue(self, task: EmbeddingTask) -> None:
        self._pending.append(task)
        self._condition.notify_all()
        logger.info(
            "bge-m3_inference task queued queue_size=%s queue_capacity=%s estimated_tokens=%s dense=%s sparse=%s colbert=%s",
            len(self._pending),
            self._settings.queue_max_items,
            task.token_estimate,
            task.signature.dense,
            task.signature.sparse,
            task.signature.colbert,
        )

    def collect_batch(self, *, stop_event: threading.Event) -> list[EmbeddingTask]:
        while not self._pending and not stop_event.is_set():
            self._condition.wait(timeout=0.1)
        if stop_event.is_set():
            return []

        first_task = self._pending[0]
        signature = first_task.signature
        deadline = first_task.enqueued_at + (self._settings.batch_max_wait_ms / 1000.0)
        while True:
            indexes = self._select_batch_indexes(signature)
            token_count = self._selected_token_count(indexes)
            max_items = self.effective_max_items(signature)
            max_tokens = self.effective_max_tokens(signature)
            should_flush = len(indexes) >= max_items or token_count >= max_tokens
            if should_flush or time.monotonic() >= deadline:
                return self._pop_selected_batch(indexes=indexes, token_count=token_count, first_task=first_task)
            remaining = max(0.0, deadline - time.monotonic())
            self._condition.wait(timeout=remaining)
            if stop_event.is_set():
                return []

    def drain(self) -> list[EmbeddingTask]:
        tasks = list(self._pending)
        self._pending.clear()
        return tasks

    def effective_batch_size(self, *, signature: EmbeddingModeSignature, task_count: int) -> int:
        return min(task_count, self.effective_max_items(signature))

    def effective_max_items(self, signature: EmbeddingModeSignature) -> int:
        return effective_max_items(
            configured_limit=self._settings.batch_max_items,
            adaptive_limit=self._adaptive_batch_items.get(signature),
            signature=signature,
        )

    def effective_max_tokens(self, signature: EmbeddingModeSignature) -> int:
        return effective_max_tokens(
            configured_limit=self._settings.batch_max_tokens,
            signature=signature,
        )

    def reduce_batch_size(self, *, signature: EmbeddingModeSignature, previous_batch_size: int) -> int:
        reduced_batch_size = max(1, previous_batch_size // 2)
        self._adaptive_batch_items[signature] = reduced_batch_size
        return reduced_batch_size

    def snapshot_signatures(self) -> list[EmbeddingModeSignature]:
        return [task.signature for task in self._pending]

    def snapshot_token_estimates(self) -> list[int]:
        return [task.token_estimate for task in self._pending]

    def _select_batch_indexes(self, signature: EmbeddingModeSignature) -> list[int]:
        return select_batch_indexes(
            task_signatures=self.snapshot_signatures(),
            token_estimates=self.snapshot_token_estimates(),
            target_signature=signature,
            max_items=self.effective_max_items(signature),
            max_tokens=self.effective_max_tokens(signature),
        )

    def _selected_token_count(self, indexes: list[int]) -> int:
        return selected_token_count(
            token_estimates=self.snapshot_token_estimates(),
            indexes=indexes,
        )

    def _pop_selected_batch(
        self,
        *,
        indexes: list[int],
        token_count: int,
        first_task: EmbeddingTask,
    ) -> list[EmbeddingTask]:
        batch = [self._pending[index] for index in indexes]
        for index in reversed(indexes):
            del self._pending[index]
        oldest_wait_ms = max(0.0, (time.monotonic() - first_task.enqueued_at) * 1000.0)
        logger.info(
            "bge-m3_inference batch dispatched batch_items=%s queue_remaining=%s estimated_tokens=%s wait_ms=%.2f dense=%s sparse=%s colbert=%s",
            len(batch),
            len(self._pending),
            token_count,
            oldest_wait_ms,
            first_task.signature.dense,
            first_task.signature.sparse,
            first_task.signature.colbert,
        )
        return batch
