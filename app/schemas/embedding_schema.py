from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator

from app.domain.config import BGE_M3_MODEL_ALIASES


EmbeddingTextInput = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EmbeddingRequestSchema(BaseModel):
    model: str = Field(min_length=1)
    input: EmbeddingTextInput | list[EmbeddingTextInput] = Field()
    dense: bool = False
    sparse: bool = False
    colbert: bool = False

    @model_validator(mode="after")
    def validate_request(self) -> "EmbeddingRequestSchema":
        if self.model not in BGE_M3_MODEL_ALIASES:
            raise ValueError("Unsupported embedding model")
        if not self.dense and not self.sparse and not self.colbert:
            raise ValueError("At least one embedding mode must be enabled")
        if isinstance(self.input, list):
            if not self.input:
                raise ValueError("At least one input is required")
            if len(self.input) > 256:
                raise ValueError("A maximum of 256 inputs is supported")
        return self


class SparseEmbeddingRead(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class EmbeddingItemRead(BaseModel):
    index: int = Field(ge=0)
    embedding: list[float] | None = None
    sparse_embedding: SparseEmbeddingRead | None = None
    colbert_embedding: list[list[float]] | None = None


class EmbeddingResponseRead(BaseModel):
    data: list[EmbeddingItemRead]


class InternalServiceHealthRead(BaseModel):
    service: str
    status: str
