from __future__ import annotations

from concurrent.futures import Future, TimeoutError
import logging
import threading
import time

from app.domain.batching import EmbeddingModeSignature
from app.domain.config import CANONICAL_BGE_M3_MODEL_NAME, CANONICAL_SERVICE_NAME
from app.domain.embedding_runtime_vectors import normalize_embedding_text
from app.domain.runtime_state import EmbeddingRuntimeState
from app.errors import (
    EmbeddingModelError,
    EmbeddingOverloadedError,
    EmbeddingRequestTimeoutError,
    EmbeddingRuntimeNotReadyError,
    EmbeddingStoppingError,
)
from app.schemas.embedding_schema import SparseEmbeddingRead

from .batch_executor import EmbeddingBatchExecutor
from .models import EmbeddingResult, EmbeddingTask
from .queue_manager import EmbeddingTaskQueue
from .runtime import FlagEmbeddingRuntime
from .settings import build_flag_embedding_client_settings


logger = logging.getLogger(__name__)


class FlagEmbeddingClient:
    def __init__(self, model_name: str = CANONICAL_BGE_M3_MODEL_NAME) -> None:
        self.model_name = model_name
        self.settings = build_flag_embedding_client_settings(model_name=model_name)

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._warmup_thread: threading.Thread | None = None
        self._load_error: EmbeddingModelError | None = None
        self._warmup_started = False
        self._worker_started = False
        self._state = EmbeddingRuntimeState.STARTING

        self._runtime = FlagEmbeddingRuntime(
            model_name=self.settings.model_name,
            model_source=self.settings.model_source,
            max_length=self.settings.max_length,
            use_fp16=self.settings.use_fp16,
        )
        self._task_queue = EmbeddingTaskQueue(condition=self._condition, settings=self.settings)
        self._batch_executor = EmbeddingBatchExecutor(runtime=self._runtime, task_queue=self._task_queue)

    def start_warmup(self) -> None:
        with self._condition:
            if self._state in {EmbeddingRuntimeState.STOPPING, EmbeddingRuntimeState.STOPPED}:
                return
            self._start_worker_locked()
            if self._warmup_started:
                return
            self._warmup_started = True
            self._transition_state_locked(EmbeddingRuntimeState.STARTING, reason="warmup_started")
            self._warmup_thread = threading.Thread(
                target=self._warmup_runtime,
                name="bge-m3-runtime-warmup",
                daemon=True,
            )
            self._warmup_thread.start()

    def stop(self) -> None:
        with self._condition:
            if self._state == EmbeddingRuntimeState.STOPPED:
                return
            self._transition_state_locked(EmbeddingRuntimeState.STOPPING, reason="shutdown_requested")
            self._stop_event.set()
            pending_tasks = self._task_queue.drain()
            self._condition.notify_all()

        self._batch_executor.fail_tasks(pending_tasks, EmbeddingStoppingError())
        self._join_thread(self._worker_thread, name="worker")
        self._join_thread(self._warmup_thread, name="warmup")

        with self._condition:
            self._transition_state_locked(EmbeddingRuntimeState.STOPPED, reason="shutdown_completed")
            self._condition.notify_all()

    def encode(
        self,
        *,
        texts: list[str],
        dense: bool,
        sparse: bool,
        colbert: bool,
    ) -> tuple[list[list[float] | None], list[SparseEmbeddingRead | None], list[list[list[float]] | None]]:
        normalized_texts = [normalize_embedding_text(text) for text in texts]
        if not normalized_texts:
            return [], [], []

        self.start_warmup()
        self._ensure_serving_ready(require_queue_capacity=False)

        signature = EmbeddingModeSignature(dense=dense, sparse=sparse, colbert=colbert)
        futures = [self._submit_task(text, signature) for text in normalized_texts]
        deadline = time.monotonic() + self.settings.request_timeout_seconds
        return self._collect_results(futures=futures, deadline=deadline)

    def check_ready(self) -> None:
        self.start_warmup()
        self._ensure_serving_ready(require_queue_capacity=True)

    def _collect_results(
        self,
        *,
        futures: list[Future[EmbeddingResult]],
        deadline: float,
    ) -> tuple[list[list[float] | None], list[SparseEmbeddingRead | None], list[list[list[float]] | None]]:
        dense_vectors: list[list[float] | None] = []
        sparse_vectors: list[SparseEmbeddingRead | None] = []
        colbert_vectors: list[list[list[float]] | None] = []

        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._cancel_futures(futures)
                raise EmbeddingRequestTimeoutError()
            try:
                dense_vector, sparse_vector, colbert_vector = future.result(timeout=remaining)
            except TimeoutError as exception:
                self._cancel_futures(futures)
                raise EmbeddingRequestTimeoutError() from exception
            dense_vectors.append(dense_vector)
            sparse_vectors.append(sparse_vector)
            colbert_vectors.append(colbert_vector)
        return dense_vectors, sparse_vectors, colbert_vectors

    def _cancel_futures(self, futures: list[Future[EmbeddingResult]]) -> None:
        for future in futures:
            future.cancel()

    def _submit_task(
        self,
        normalized_text: str,
        signature: EmbeddingModeSignature,
    ) -> Future[EmbeddingResult]:
        token_estimate = self._runtime.estimate_tokens(normalized_text)
        future: Future[EmbeddingResult] = Future()
        task = EmbeddingTask(
            normalized_text=normalized_text,
            signature=signature,
            token_estimate=token_estimate,
            future=future,
            enqueued_at=time.monotonic(),
        )
        with self._condition:
            self._ensure_serving_ready_locked(require_queue_capacity=True)
            self._task_queue.enqueue(task)
        return future

    def _start_worker_locked(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="bge-m3-batch-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _warmup_runtime(self) -> None:
        try:
            self._runtime.ensure_loaded()
        except EmbeddingModelError as exception:
            with self._condition:
                self._load_error = exception
                self._transition_state_locked(EmbeddingRuntimeState.DEGRADED, reason="warmup_failed")
        else:
            with self._condition:
                if self._state not in {EmbeddingRuntimeState.STOPPING, EmbeddingRuntimeState.STOPPED}:
                    self._transition_state_locked(EmbeddingRuntimeState.READY, reason="warmup_completed")
        finally:
            self._ready_event.set()
            with self._condition:
                self._condition.notify_all()

    def _worker_loop(self) -> None:
        logger.info("bge-m3_inference worker started")
        while not self._stop_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            try:
                self._process_batch(batch)
            except Exception as exception:  # pragma: no cover - defensive path
                logger.exception("bge-m3_inference batch failed")
                error = exception if isinstance(exception, EmbeddingModelError) else EmbeddingModelError(
                    "Unable to encode embeddings"
                )
                self._batch_executor.fail_tasks(batch, error)
        logger.info("bge-m3_inference worker stopped")

    def _collect_batch(self) -> list[EmbeddingTask]:
        with self._condition:
            return self._task_queue.collect_batch(stop_event=self._stop_event)

    def _process_batch(self, batch: list[EmbeddingTask]) -> None:
        self._ensure_runtime_loaded_for_batch()
        self._batch_executor.process_batch(batch)

    def _ensure_serving_ready(self, *, require_queue_capacity: bool) -> None:
        with self._condition:
            self._ensure_serving_ready_locked(require_queue_capacity=require_queue_capacity)

    def _ensure_serving_ready_locked(self, *, require_queue_capacity: bool) -> None:
        if self._state in {EmbeddingRuntimeState.STOPPING, EmbeddingRuntimeState.STOPPED} or self._stop_event.is_set():
            raise EmbeddingStoppingError()
        if not self._ready_event.is_set():
            raise EmbeddingRuntimeNotReadyError()
        if self._load_error is not None:
            self._transition_state_locked(EmbeddingRuntimeState.DEGRADED, reason="load_error_present")
            raise self._load_error
        if not self._worker_is_alive():
            self._transition_state_locked(EmbeddingRuntimeState.DEGRADED, reason="worker_not_alive")
            raise EmbeddingModelError(f"{CANONICAL_SERVICE_NAME} batch worker is not running")
        if require_queue_capacity and self._task_queue.is_full():
            raise EmbeddingOverloadedError()
        if self._state == EmbeddingRuntimeState.STARTING:
            self._transition_state_locked(EmbeddingRuntimeState.READY, reason="runtime_ready")

    def _ensure_runtime_loaded_for_batch(self) -> None:
        if self._load_error is not None:
            raise self._load_error
        if not self._ready_event.is_set():
            raise EmbeddingRuntimeNotReadyError()

    def _worker_is_alive(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _join_thread(self, thread: threading.Thread | None, *, name: str) -> None:
        if thread is None:
            return
        thread.join(timeout=self.settings.shutdown_grace_seconds)
        if thread.is_alive():
            logger.warning(
                "bge-m3_inference %s join timed out shutdown_grace_s=%.2f",
                name,
                self.settings.shutdown_grace_seconds,
            )

    def _transition_state_locked(self, new_state: EmbeddingRuntimeState, *, reason: str) -> None:
        if self._state == new_state:
            return
        previous_state = self._state
        self._state = new_state
        logger.info(
            "bge-m3_inference state transition previous=%s current=%s reason=%s queue_size=%s",
            previous_state.value,
            new_state.value,
            reason,
            self._task_queue.size(),
        )
