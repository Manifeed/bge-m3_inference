from __future__ import annotations


class EmbeddingServiceError(RuntimeError):
    """Base class for stable embedding service errors exposed over HTTP."""

    code = "embedding_model_unavailable"
    default_message = "Unable to serve embeddings"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class EmbeddingServiceConfigurationError(EmbeddingServiceError):
    code = "embedding_service_misconfigured"
    default_message = "Embedding service configuration is invalid"


class EmbeddingRuntimeNotReadyError(EmbeddingServiceError):
    code = "warming_up"
    default_message = "bge-m3_inference is still warming up"


class EmbeddingOverloadedError(EmbeddingServiceError):
    code = "overloaded"
    default_message = "bge-m3_inference queue is full"


class EmbeddingStoppingError(EmbeddingServiceError):
    code = "stopping"
    default_message = "bge-m3_inference is stopping"


class EmbeddingRequestTimeoutError(EmbeddingServiceError):
    code = "request_timeout"
    default_message = "bge-m3_inference request timed out"


class EmbeddingModelError(EmbeddingServiceError):
    code = "embedding_model_unavailable"
    default_message = "Unable to serve embeddings"
