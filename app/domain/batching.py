from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModeSignature:
    dense: bool
    sparse: bool
    colbert: bool


def effective_max_items(
    *,
    configured_limit: int,
    adaptive_limit: int | None,
    signature: EmbeddingModeSignature,
) -> int:
    resolved_limit = adaptive_limit if adaptive_limit is not None else configured_limit
    if signature.colbert:
        resolved_limit = max(1, resolved_limit // 2)
    return max(1, resolved_limit)


def effective_max_tokens(*, configured_limit: int, signature: EmbeddingModeSignature) -> int:
    if signature.colbert:
        return max(1, configured_limit // 2)
    return max(1, configured_limit)
