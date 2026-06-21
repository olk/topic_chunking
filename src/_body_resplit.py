# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT

"""Body-level text resplitting for DoclingTopicChunking.

Splits a body string into sub-bodies that each fit within a token cap.
Used when the heading prefix consumes part of the per-chunk budget, leaving
only ``max_tokens - heading_tokens`` for the body.

Returns (sub_text, start_offset, end_offset) triples so the caller can
map each sub-body back to source sentence provenance.

The function uses spaCy ``nlp(body).sents`` as the FIRST segmentation step
so that the token cap is enforced at sentence boundaries only. Sentences
that themselves exceed the cap are emitted intact as a single sub-body
(atomic emission — cap is best-effort; no mid-sentence slicing).

This mirrors the main path of :class:`TopicChunking`, whose chunks are always
built from whole sentences joined with " " (see :func:`src.topic.chunk`).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resplit_body(
    body: str,
    cap_tokens: int,
    tokenizer: Any,
    nlp: Any,
) -> list[tuple[str, int, int]]:
    """Split body into sub-bodies each ≤ cap_tokens, never cutting inside a sentence.

    Uses spaCy sentence segmentation as the first step. Sentences that
    individually fit are packed greedily into sub-bodies. A single sentence
    that itself exceeds ``cap_tokens`` is emitted intact as its own sub-body
    (cap is best-effort).

    Args:
        body: The body text to split.
        cap_tokens: Maximum tokens per sub-body.
        tokenizer: A tokenizer implementing ``encode(text) -> list[int]``.
        nlp: A spaCy pipeline (Language) used to segment ``body`` into sentences.

    Returns:
        List of (sub_text, start_offset, end_offset) where start_offset and
        end_offset are character positions in the original ``body``.
    """
    if cap_tokens <= 0:
        raise ValueError("cap_tokens must be positive")
    if not body or not body.strip():
        return []

    sentence_units: list[tuple[str, int, int]] = []
    doc = nlp(body)
    for sent in doc.sents:
        txt = sent.text.strip()
        if not txt:
            continue
        cs = sent.start_char
        ce = cs + len(txt)
        sentence_units.append((txt, cs, ce))

    if not sentence_units:
        sentence_units = [(body.strip(), 0, len(body))]

    out: list[tuple[str, int, int]] = []
    buffer_texts: list[str] = []
    buffer_starts: list[int] = []
    buffer_ends: list[int] = []
    buffer_tokens = 0

    def _flush() -> None:
        if not buffer_texts:
            return
        sub_text = " ".join(buffer_texts)
        out.append((sub_text, buffer_starts[0], buffer_ends[-1]))

    for sent_text, sent_start, sent_end in sentence_units:
        sent_tokens = len(tokenizer.encode(sent_text))
        if sent_tokens > cap_tokens:
            _flush()
            buffer_texts = []
            buffer_starts = []
            buffer_ends = []
            buffer_tokens = 0
            logger.warning(
                "Sentence (%d tokens) exceeds cap (%d); emitting atomic "
                "without splitting — cap is best-effort",
                sent_tokens, cap_tokens,
            )
            out.append((sent_text, sent_start, sent_end))
            continue

        if buffer_texts and buffer_tokens + sent_tokens > cap_tokens:
            _flush()
            buffer_texts = []
            buffer_starts = []
            buffer_ends = []
            buffer_tokens = 0

        if not buffer_texts:
            buffer_starts.append(sent_start)
        buffer_texts.append(sent_text)
        buffer_ends.append(sent_end)
        buffer_tokens += sent_tokens

    _flush()

    return out
