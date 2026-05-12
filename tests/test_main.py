from __future__ import annotations

from fastapi.testclient import TestClient

from app.errors import (
    EmbeddingModelError,
    EmbeddingOverloadedError,
    EmbeddingRequestTimeoutError,
    EmbeddingRuntimeNotReadyError,
    EmbeddingStoppingError,
)
from app.main import app
from app.schemas.embedding_schema import EmbeddingResponseRead, SparseEmbeddingRead


def _patch_runtime_client(monkeypatch) -> None:
    class DummyRuntimeClient:
        def start_warmup(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr("app.main.get_flag_embedding_client", lambda: DummyRuntimeClient())


def test_internal_ready_returns_503_when_embedding_runtime_is_warming_up(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.main.check_embedding_model_ready",
        lambda: (_ for _ in ()).throw(EmbeddingRuntimeNotReadyError()),
    )

    client = TestClient(app)
    response = client.get("/internal/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "warming_up",
        "message": "bge-m3_inference is still warming up",
    }


def test_internal_ready_returns_canonical_service_name(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr("app.main.check_embedding_model_ready", lambda: None)

    client = TestClient(app)
    response = client.get("/internal/ready")

    assert response.status_code == 200
    assert response.json() == {
        "service": "bge-m3_inference",
        "status": "ready",
    }


def test_embeddings_route_returns_503_when_embedding_runtime_is_unavailable(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.routers.embedding_router.create_embeddings",
        lambda payload: (_ for _ in ()).throw(EmbeddingModelError("Unable to load embedding model")),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": True,
            "sparse": True,
            "colbert": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "embedding_model_unavailable",
        "message": "Unable to load embedding model",
    }


def test_embeddings_route_returns_503_when_queue_is_overloaded(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.routers.embedding_router.create_embeddings",
        lambda payload: (_ for _ in ()).throw(EmbeddingOverloadedError()),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "overloaded",
        "message": "bge-m3_inference queue is full",
    }


def test_embeddings_route_returns_503_when_service_is_stopping(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.routers.embedding_router.create_embeddings",
        lambda payload: (_ for _ in ()).throw(EmbeddingStoppingError()),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "stopping",
        "message": "bge-m3_inference is stopping",
    }


def test_embeddings_route_returns_503_when_internal_wait_times_out(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.routers.embedding_router.create_embeddings",
        lambda payload: (_ for _ in ()).throw(EmbeddingRequestTimeoutError()),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "request_timeout",
        "message": "bge-m3_inference request timed out",
    }


def test_embeddings_route_returns_payload_when_embedding_runtime_is_ready(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.routers.embedding_router.create_embeddings",
        lambda payload: EmbeddingResponseRead(
            data=[
                {
                    "index": 0,
                    "embedding": None,
                    "sparse_embedding": None,
                    "colbert_embedding": [[0.1, 0.2], [0.3, 0.4]],
                }
            ],
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": False,
            "sparse": False,
            "colbert": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {
                "index": 0,
                "embedding": None,
                "sparse_embedding": None,
                "colbert_embedding": [[0.1, 0.2], [0.3, 0.4]],
            }
        ],
    }


def test_embeddings_route_trims_edges_without_collapsing_internal_spacing(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")
    seen: dict[str, str | list[str]] = {}

    def capture_payload(payload) -> EmbeddingResponseRead:
        seen["input"] = payload.input
        return EmbeddingResponseRead(
            data=[
                {
                    "index": 0,
                    "embedding": [0.1],
                    "sparse_embedding": None,
                    "colbert_embedding": None,
                }
            ],
        )

    monkeypatch.setattr("app.routers.embedding_router.create_embeddings", capture_payload)

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["  hello   world\nnext line  "],
            "dense": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert seen["input"] == ["hello   world\nnext line"]


def test_embeddings_route_rejects_request_without_any_embedding_mode(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422
    assert "At least one embedding mode must be enabled" in response.text


def test_embeddings_route_rejects_empty_input_list(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": [],
            "dense": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422
    assert "At least one input is required" in response.text


def test_embeddings_route_rejects_blank_input(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["   "],
            "dense": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422


def test_embeddings_route_requires_api_key(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.setenv("BGE_M3_INFERENCE_API_KEY", "test-key")

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": True,
            "sparse": True,
            "colbert": False,
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_embeddings_route_accepts_legacy_api_key_env(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.delenv("BGE_M3_INFERENCE_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_SERVICE_API_KEY", "legacy-key")
    monkeypatch.setattr(
        "app.routers.embedding_router.create_embeddings",
        lambda payload: EmbeddingResponseRead(
            data=[
                {
                    "index": 0,
                    "embedding": [0.1, 0.2],
                    "sparse_embedding": SparseEmbeddingRead(indices=[1], values=[0.3]),
                    "colbert_embedding": None,
                }
            ],
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": True,
            "sparse": True,
            "colbert": False,
        },
        headers={"Authorization": "Bearer legacy-key"},
    )

    assert response.status_code == 200


def test_internal_ready_requires_api_key_configuration(monkeypatch) -> None:
    _patch_runtime_client(monkeypatch)
    monkeypatch.delenv("BGE_M3_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_SERVICE_API_KEY", raising=False)
    client = TestClient(app)
    response = client.get("/internal/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "embedding_service_misconfigured",
        "message": "BGE_M3_INFERENCE_API_KEY is required "
        "(legacy EMBEDDING_SERVICE_API_KEY is still accepted temporarily)",
    }
