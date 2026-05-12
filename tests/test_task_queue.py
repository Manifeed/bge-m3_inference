from __future__ import annotations

from concurrent.futures import Future
import threading

from app.clients.flag_embedding_client.models import EmbeddingTask
from app.clients.flag_embedding_client.queue_manager import EmbeddingTaskQueue
from app.clients.flag_embedding_client.settings import build_flag_embedding_client_settings
from app.domain.batching import EmbeddingModeSignature


def test_select_batch_indexes_respects_signature_and_token_limit() -> None:
    settings = build_flag_embedding_client_settings()
    settings.batch_max_items = 3
    settings.batch_max_tokens = 5
    queue = EmbeddingTaskQueue(condition=threading.Condition(), settings=settings)
    target_signature = EmbeddingModeSignature(dense=True, sparse=False, colbert=False)
    other_signature = EmbeddingModeSignature(dense=False, sparse=True, colbert=False)
    queue._pending = [
        EmbeddingTask("first", target_signature, 3, Future(), 0.0),
        EmbeddingTask("other", other_signature, 1, Future(), 0.0),
        EmbeddingTask("second", target_signature, 2, Future(), 0.0),
        EmbeddingTask("third", target_signature, 4, Future(), 0.0),
    ]

    indexes, token_count = queue._select_batch(target_signature)

    assert indexes == [0, 2]
    assert token_count == 5


def test_collect_batch_skips_cancelled_tasks() -> None:
    settings = build_flag_embedding_client_settings()
    settings.batch_max_wait_ms = 1
    queue = EmbeddingTaskQueue(condition=threading.Condition(), settings=settings)
    signature = EmbeddingModeSignature(dense=True, sparse=False, colbert=False)

    cancelled_future = Future()
    cancelled_future.cancel()
    active_future = Future()
    queue._pending = [
        EmbeddingTask("cancelled", signature, 1, cancelled_future, 0.0),
        EmbeddingTask("active", signature, 2, active_future, 0.0),
    ]

    batch = queue.collect_batch(stop_event=threading.Event())

    assert [task.normalized_text for task in batch] == ["active"]
    assert queue.size() == 0
