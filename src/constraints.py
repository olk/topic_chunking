# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT

"""Cross-field token-limit constraints for the RAG pipeline.

The RAG server ingests German PDFs through a chain of components that each
impose a token cap:

  chunker  --max_tokens-->  embedder  --max_embedder_tokens-->  vector store
                            |
                            +--max_entity_tokens-->  LLM entity extraction

If any chunker's ``max_tokens`` is larger than the next stage's cap, the
oversized chunk is silently truncated before being embedded or sent to the
LLM.  That destroys information and violates the project invariant
"no truncation of chunked data by embedding or LLM".

This module provides the runtime safety net for that invariant:
:func:`resplit_oversized_chunks` re-splits any chunk that exceeds the cap,
instead of letting the embedder or LLM truncate it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Sequence

from .models import Chunk
from .tokenizer import count_tokens

if TYPE_CHECKING:
    from docling_core.transforms.chunker import BaseChunk

logger = logging.getLogger(__name__)


def resplit_oversized_chunks(
    chunks: Sequence[Any],
    cap_tokens: int,
    tokenizer: Any,
    sentence_splitter: Callable[[str], list[str]] | None = None,
) -> List[Chunk]:
    """Re-split any chunk whose token count exceeds ``cap_tokens``.

    Splits oversized chunks into atomic units (lines, then whitespace-separated
    tokens as a fallback) and greedily packs them back into sub-chunks.  We
    never truncate, so the total information is preserved.

    If ``sentence_splitter`` is provided, it is used to split oversized units
    into sentences before falling back to token-level slicing.  This preserves
    sentence boundaries for natural language text.
    """
    if cap_tokens <= 0:
        raise ValueError("cap_tokens must be positive")
    if not chunks:
        return []

    def _split_into_sentences(unit: str) -> list[str]:
        if not sentence_splitter:
            return [unit]
        sentences = sentence_splitter(unit)
        return [s for s in sentences if s.strip()] or [unit]

    def _start_char(c: Any) -> int:
        span = getattr(c, "charspan", None)
        if span:
            return span[0]
        return getattr(c, "start_char", 0)

    def _end_char(c: Any) -> int:
        span = getattr(c, "charspan", None)
        if span:
            return span[1]
        return getattr(c, "end_char", 0)

    out: List[Chunk] = []
    next_index = 0
    for chunk in chunks:
        text = chunk.text or ""
        if not text.strip():
            continue
        token_count = count_tokens(text, tokenizer)
        if token_count <= cap_tokens:
            new_chunk = Chunk(
                text=text,
                index=next_index,
                start_char=_start_char(chunk),
                end_char=_end_char(chunk),
                metadata=dict(getattr(chunk, "metadata", None) or {}),
            )
            out.append(new_chunk)
            next_index += 1
            continue

        base_start = _start_char(chunk)
        metadata = dict(getattr(chunk, "metadata", None) or {})

        units: List[str] = []
        if "\n" in text:
            for line in text.splitlines(keepends=True):
                if line:
                    units.append(line)
        else:
            i = 0
            n = len(text)
            while i < n:
                j = i
                while j < n and text[j].isspace():
                    j += 1
                if j > i:
                    units.append(text[i:j])
                    i = j
                k = i
                while k < n and not text[k].isspace():
                    k += 1
                if k > i:
                    units.append(text[i:k])
                    i = k

        buffer: List[str] = []
        buffer_tokens = 0
        for unit in units:
            unit_tokens = count_tokens(unit, tokenizer)
            if unit_tokens > cap_tokens:
                if buffer:
                    out.append(
                        Chunk(
                            text="".join(buffer),
                            index=next_index,
                            start_char=base_start,
                            end_char=base_start + sum(len(u) for u in buffer),
                            metadata=dict(metadata),
                        )
                    )
                    next_index += 1
                    buffer = []
                    buffer_tokens = 0
                sub_units = _split_into_sentences(unit)
                if len(sub_units) > 1:
                    for s in sub_units:
                        s_tokens = count_tokens(s, tokenizer)
                        if buffer and buffer_tokens + s_tokens > cap_tokens:
                            out.append(
                                Chunk(
                                    text="".join(buffer),
                                    index=next_index,
                                    start_char=base_start,
                                    end_char=base_start + sum(len(u) for u in buffer),
                                    metadata=dict(metadata),
                                )
                            )
                            next_index += 1
                            buffer = []
                            buffer_tokens = 0
                        buffer.append(s)
                        buffer_tokens += s_tokens
                    continue
                encoded = tokenizer.encode(unit)
                if hasattr(encoded, "__len__"):
                    n_tokens = len(encoded)
                else:
                    n_tokens = sum(1 for _ in encoded)
                cursor = 0
                for tok_idx in range(0, n_tokens, cap_tokens):
                    sub_encoded = encoded[tok_idx:tok_idx + cap_tokens]
                    if hasattr(tokenizer, "decode"):
                        sub_text = tokenizer.decode(list(sub_encoded))
                    else:
                        sub_text = unit
                    if not sub_text:
                        continue
                    out.append(
                        Chunk(
                            text=sub_text,
                            index=next_index,
                            start_char=base_start + cursor,
                            end_char=base_start + cursor + len(sub_text),
                            metadata=dict(metadata),
                        )
                    )
                    next_index += 1
                    cursor += len(sub_text)
                continue

            if buffer and buffer_tokens + unit_tokens > cap_tokens:
                out.append(
                    Chunk(
                        text="".join(buffer),
                        index=next_index,
                        start_char=base_start,
                        end_char=base_start + sum(len(u) for u in buffer),
                        metadata=dict(metadata),
                    )
                )
                next_index += 1
                buffer = []
                buffer_tokens = 0
            buffer.append(unit)
            buffer_tokens += unit_tokens

        if buffer:
            sub_text = "".join(buffer)
            out.append(
                Chunk(
                    text=sub_text,
                    index=next_index,
                    start_char=base_start,
                    end_char=base_start + len(sub_text),
                    metadata=dict(metadata),
                )
            )
            next_index += 1

    return out


def resplit_oversized_doc_chunks(
    chunks: Sequence["BaseChunk"],
    cap_tokens: int,
    tokenizer: Any,
    sentence_splitter: Callable[[str], list[str]] | None = None,
) -> list["BaseChunk"]:
    """Re-split any DocChunk whose token count exceeds ``cap_tokens``.

    Preserves the docling ``meta`` (doc_items, headings, origin, source_charspan,
    source_pages, source_bboxes) by partitioning at doc_item boundaries.
    """
    if cap_tokens <= 0:
        raise ValueError("cap_tokens must be positive")
    if not chunks:
        return []

    def _split_into_sentences(unit: str) -> list[str]:
        if not sentence_splitter:
            return [unit]
        return [s for s in sentence_splitter(unit) if s.strip()] or [unit]

    def _doc_item_charspan(item: Any) -> tuple[int, int] | None:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            return prov[0].charspan
        return None

    out: list["BaseChunk"] = []
    next_index = 0

    for chunk in chunks:
        from docling_core.transforms.chunker.doc_chunk import DocChunk

        text = chunk.text or ""
        if not text.strip():
            continue

        token_count = count_tokens(text, tokenizer)
        if token_count <= cap_tokens:
            out.append(chunk)
            next_index += 1
            continue

        doc_items = getattr(chunk.meta, "doc_items", []) if hasattr(chunk, "meta") else []

        units: list[str] = []
        if "\n" in text:
            for line in text.splitlines(keepends=True):
                if line:
                    units.append(line)
        else:
            i = 0
            n = len(text)
            while i < n:
                j = i
                while j < n and text[j].isspace():
                    j += 1
                if j > i:
                    units.append(text[i:j])
                    i = j
                k = i
                while k < n and not text[k].isspace():
                    k += 1
                if k > i:
                    units.append(text[i:k])
                    i = k

        def emit(sub_text: str) -> None:
            out.append(
                DocChunk(
                    text=sub_text,
                    meta=chunk.meta,  # type: ignore[arg-type]
                )
            )
            nonlocal next_index
            next_index += 1

        buffer: list[str] = []
        buffer_tokens = 0
        for unit in units:
            unit_tokens = count_tokens(unit, tokenizer)
            if unit_tokens > cap_tokens:
                if buffer:
                    emit("".join(buffer))
                    buffer = []
                    buffer_tokens = 0
                sub_units = _split_into_sentences(unit)
                if len(sub_units) > 1:
                    for s in sub_units:
                        s_tokens = count_tokens(s, tokenizer)
                        if buffer and buffer_tokens + s_tokens > cap_tokens:
                            emit("".join(buffer))
                            buffer = []
                            buffer_tokens = 0
                        buffer.append(s)
                        buffer_tokens += s_tokens
                    continue
                encoded = tokenizer.encode(unit)
                if hasattr(encoded, "__len__"):
                    n_tokens = len(encoded)
                else:
                    n_tokens = sum(1 for _ in encoded)
                for tok_idx in range(0, n_tokens, cap_tokens):
                    sub_encoded = encoded[tok_idx:tok_idx + cap_tokens]
                    if hasattr(tokenizer, "decode"):
                        sub_text = tokenizer.decode(list(sub_encoded))
                    else:
                        sub_text = unit
                    if sub_text:
                        emit(sub_text)
                continue

            if buffer and buffer_tokens + unit_tokens > cap_tokens:
                emit("".join(buffer))
                buffer = []
                buffer_tokens = 0
            buffer.append(unit)
            buffer_tokens += unit_tokens

        if buffer:
            emit("".join(buffer))

    return out
