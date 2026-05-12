from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


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


def select_batch_indexes(
    *,
    task_signatures: Sequence[EmbeddingModeSignature],
    token_estimates: Sequence[int],
    target_signature: EmbeddingModeSignature,
    max_items: int,
    max_tokens: int,
) -> list[int]:
    indexes: list[int] = []
    token_count = 0
    for index, signature in enumerate(task_signatures):
        if signature != target_signature:
            continue
        next_token_count = token_count + token_estimates[index]
        if indexes and next_token_count > max_tokens:
            break
        indexes.append(index)
        token_count = next_token_count
        if len(indexes) >= max_items:
            break
    return indexes or [0]


def selected_token_count(*, token_estimates: Sequence[int], indexes: Sequence[int]) -> int:
    return sum(token_estimates[index] for index in indexes if index < len(token_estimates))
