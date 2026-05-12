from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from app.clients.flag_embedding_client import EmbeddingModelError
from app.clients.flag_embedding_client.models import EmbeddingTask
from app.domain.batching import EmbeddingModeSignature
from app.domain.runtime_state import EmbeddingRuntimeState
from app.errors import EmbeddingOverloadedError, EmbeddingRequestTimeoutError, EmbeddingStoppingError


def _mark_client_ready(client) -> None:
    client._ready_event.set()
    client._state = EmbeddingRuntimeState.READY
    client._worker_thread = SimpleNamespace(is_alive=lambda: True)


def test_submit_task_rejects_when_queue_is_full() -> None:
    from app.clients.flag_embedding_client.client import FlagEmbeddingClient

    client = FlagEmbeddingClient()
    _mark_client_ready(client)
    client.settings.queue_max_items = 1
    signature = EmbeddingModeSignature(dense=True, sparse=False, colbert=False)
    client._task_queue._pending = [
        EmbeddingTask("occupied", signature, 1, Future(), 0.0),
    ]

    with pytest.raises(EmbeddingOverloadedError):
        client._submit_task("hello", signature)


def test_check_ready_rejects_when_worker_is_not_alive() -> None:
    from app.clients.flag_embedding_client.client import FlagEmbeddingClient

    client = FlagEmbeddingClient()
    client._ready_event.set()
    client._state = EmbeddingRuntimeState.READY
    client._worker_thread = SimpleNamespace(is_alive=lambda: False)
    client.start_warmup = lambda: None

    with pytest.raises(EmbeddingModelError, match="batch worker is not running"):
        client.check_ready()


def test_encode_raises_request_timeout_when_futures_take_too_long(monkeypatch) -> None:
    from app.clients.flag_embedding_client.client import FlagEmbeddingClient

    client = FlagEmbeddingClient()
    _mark_client_ready(client)
    client.settings.request_timeout_seconds = 0.01
    signature = EmbeddingModeSignature(dense=True, sparse=False, colbert=False)

    monkeypatch.setattr(client, "start_warmup", lambda: None)
    monkeypatch.setattr(client, "_submit_task", lambda normalized_text, signature: Future())

    with pytest.raises(EmbeddingRequestTimeoutError):
        client.encode(texts=["hello"], dense=signature.dense, sparse=signature.sparse, colbert=signature.colbert)


def test_stop_drains_pending_tasks_with_stopping_error() -> None:
    from app.clients.flag_embedding_client.client import FlagEmbeddingClient

    client = FlagEmbeddingClient()
    future = Future()
    signature = EmbeddingModeSignature(dense=True, sparse=False, colbert=False)
    client._task_queue._pending = [
        EmbeddingTask("hello", signature, 1, future, 0.0),
    ]

    client.stop()

    assert isinstance(future.exception(), EmbeddingStoppingError)
    assert client._state == EmbeddingRuntimeState.STOPPED


def test_process_batch_retries_with_smaller_batch_after_cuda_oom(monkeypatch) -> None:
    from app.clients.flag_embedding_client.client import FlagEmbeddingClient

    client = FlagEmbeddingClient()
    client._ready_event.set()
    signature = EmbeddingModeSignature(dense=True, sparse=False, colbert=False)
    futures = [Future(), Future(), Future(), Future()]
    batch = [
        EmbeddingTask(f"text-{index}", signature, 4, future, 0.0)
        for index, future in enumerate(futures)
    ]
    calls: list[int] = []

    def fake_encode_batch(*, texts, signature, batch_size):
        calls.append(batch_size)
        if len(calls) == 1:
            raise EmbeddingModelError("CUDA out of memory")
        return (
            [[float(index)] for index, _ in enumerate(texts)],
            [None for _ in texts],
            [None for _ in texts],
        )

    monkeypatch.setattr(client._runtime, "encode_batch", fake_encode_batch)

    client._process_batch(batch)

    assert calls == [4, 2]
    assert client._task_queue._adaptive_batch_items[signature] == 2
    assert futures[0].result() == ([0.0], None, None)
    assert futures[-1].result() == ([3.0], None, None)


def test_process_batch_propagates_oom_after_retry_failure(monkeypatch) -> None:
    from app.clients.flag_embedding_client.client import FlagEmbeddingClient

    client = FlagEmbeddingClient()
    client._ready_event.set()
    signature = EmbeddingModeSignature(dense=True, sparse=False, colbert=False)
    batch = [
        EmbeddingTask("text-0", signature, 4, Future(), 0.0),
        EmbeddingTask("text-1", signature, 4, Future(), 0.0),
    ]
    calls: list[int] = []

    def fake_encode_batch(*, texts, signature, batch_size):
        calls.append(batch_size)
        raise EmbeddingModelError("CUDA out of memory")

    monkeypatch.setattr(client._runtime, "encode_batch", fake_encode_batch)

    with pytest.raises(EmbeddingModelError, match="out of memory"):
        client._process_batch(batch)

    assert calls == [2, 1]
