# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT

"""Pure topic-splitting algorithm.

Provides :func:`split_by_topic`, a generic function that takes a flat list of
objects with a ``.text`` attribute and returns inclusive (start_idx, end_idx)
ranges for each topic sub-chunk.  The caller is responsible for mapping those
indices back to whatever domain object (sentence, doc_item, etc.) it needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class HasText(Protocol):
    text: str


@dataclass(frozen=True)
class TopicSplitParams:
    embed_model: Any
    breakpoint_threshold_percentile: int = 95
    min_sentences_per_chunk: int = 1
    buffer_size: int = 1


def split_by_topic(
    sentences: Sequence[Any],
    params: TopicSplitParams,
) -> List[tuple[int, int]]:
    """Return inclusive (start_idx, end_idx) ranges for each topic sub-chunk.

    Args:
        sentences: Sequence of objects with a ``.text`` attribute.
        params: Configuration for the topic-splitting algorithm.

    Returns:
        List of (start_idx, end_idx) inclusive ranges, one per detected topic
        region.  Empty list if the input is empty.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    windowed_texts = _build_sentence_groups(sentences, params.buffer_size)

    try:
        embeddings = params.embed_model.get_text_embedding_batch(windowed_texts)
    except Exception as e:
        raise ValueError(f"Topic split embedding failed: {e}") from e

    if len(embeddings) != n:
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)}, expected {n}"
        )

    distances = _cosine_distances(embeddings)

    threshold = float(
        np.percentile(distances, params.breakpoint_threshold_percentile)
    )

    breakpoints = [0]
    for i, dist in enumerate(distances):
        if dist > threshold:
            breakpoints.append(i + 1)

    ranges: List[tuple[int, int]] = []
    for idx, start_idx in enumerate(breakpoints):
        end_idx = breakpoints[idx + 1] if idx + 1 < len(breakpoints) else n
        if (
            end_idx - start_idx < params.min_sentences_per_chunk
            and len(breakpoints) > 1
        ):
            continue
        ranges.append((start_idx, end_idx - 1))

    if not ranges:
        ranges = [(0, n - 1)]

    logger.debug("Topic split: %d chunks from %d sentences", len(ranges), n)
    return ranges


def _build_sentence_groups(
    sentences: Sequence[Any], buffer_size: int
) -> List[str]:
    n = len(sentences)
    groups: List[str] = []
    for i in range(n):
        parts: List[str] = []
        for j in range(i - buffer_size, i):
            if j >= 0:
                parts.append(sentences[j].text)
        parts.append(sentences[i].text)
        for j in range(i + 1, i + 1 + buffer_size):
            if j < n:
                parts.append(sentences[j].text)
        groups.append("".join(parts))
    return groups


def _cosine_distances(embeddings: List[Any]) -> List[float]:
    distances: List[float] = []
    for i in range(len(embeddings) - 1):
        v1 = np.asarray(embeddings[i])
        v2 = np.asarray(embeddings[i + 1])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim = float(np.dot(v1, v2) / (norm1 * norm2))
        distances.append(1.0 - sim)
    return distances
