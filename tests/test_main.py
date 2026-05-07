from __future__ import annotations

from fastapi.testclient import TestClient

from app.clients.flag_embedding_client import EmbeddingModelError
from app.main import app
from app.schemas.embedding_schema import EmbeddingResponseRead, SparseEmbeddingRead


def test_internal_ready_returns_503_when_embedding_runtime_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_SERVICE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.main.check_embedding_model_ready",
        lambda: (_ for _ in ()).throw(
            EmbeddingModelError("Unable to import FlagEmbedding runtime dependencies: broken import")
        ),
    )

    client = TestClient(app)
    response = client.get("/internal/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "embedding_model_unavailable",
        "message": "Unable to import FlagEmbedding runtime dependencies: broken import",
    }


def test_embeddings_route_returns_503_when_embedding_runtime_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_SERVICE_API_KEY", "test-key")
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
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "embedding_model_unavailable",
        "message": "Unable to load embedding model",
    }


def test_embeddings_route_returns_payload_when_embedding_runtime_is_ready(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_SERVICE_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.routers.embedding_router.create_embeddings",
        lambda payload: EmbeddingResponseRead(
            data=[
                {
                    "index": 0,
                    "embedding": [0.1, 0.2],
                    "sparse_embedding": SparseEmbeddingRead(indices=[1], values=[0.3]),
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
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {
                "index": 0,
                "embedding": [0.1, 0.2],
                "sparse_embedding": {"indices": [1], "values": [0.3]},
            }
        ],
    }


def test_embeddings_route_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_SERVICE_API_KEY", "test-key")

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-m3",
            "input": ["hello"],
            "dense": True,
            "sparse": True,
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_internal_ready_requires_api_key_configuration(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_SERVICE_API_KEY", raising=False)
    client = TestClient(app)
    response = client.get("/internal/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "embedding_service_misconfigured",
        "message": "EMBEDDING_SERVICE_API_KEY is required",
    }
