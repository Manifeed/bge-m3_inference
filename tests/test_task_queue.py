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

    indexes = queue._select_batch_indexes(target_signature)

    assert indexes == [0, 2]
