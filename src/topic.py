# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT

"""Topic-based chunking strategy.

Implements Greg Kamradt's "level-3 semantic chunking" algorithm
(https://youtu.be/8OJC21T2SL4?t=1933). Sentence splitting uses spaCy's
``de_dep_news_trf`` transformer model; sentence boundaries are driven by
the dependency parser (doc.sents). The shared pipeline is loaded once via
:func:`._spacy_model.get_spacy_nlp`.
"""

import logging
from typing import Any, List

from ._spacy_model import get_spacy_nlp
from ._topic_split import TopicSplitParams, split_by_topic
from .models import Chunk
from .tokenizer import TokenizerProtocol

logger = logging.getLogger(__name__)


class _TextWithPos:
    __slots__ = ("text", "start_char", "end_char")

    def __init__(self, text: str, start_char: int, end_char: int):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class TopicChunking:
    """Topic-based chunking using embedding similarity for topic boundary detection.

    Implements Greg Kamradt's "level-3 semantic chunking" algorithm
    (https://youtu.be/8OJC21T2SL4?t=1933). Splits text by detecting semantic
    topic boundaries based on cosine distance between consecutive sentence
    embeddings (distance = ``1 - embed_model.similarity(...)``, default
    ``SimilarityMode.DEFAULT = cosine``). Uses a percentile of the distance
    distribution as the threshold; pairs with distance > threshold become
    chunk boundaries.

    The ``buffer_size`` parameter controls how many surrounding sentences are
    concatenated into each embedding window (mirrors LlamaIndex's
    ``SemanticSplitterNodeParser.buffer_size``). With ``buffer_size=1``, each
    sentence is embedded as ``[prev_sentence] + [current] + [next_sentence]``;
    boundaries are clipped to available neighbors. ``buffer_size=0`` embeds
    each sentence individually (legacy mode, pre-2.0 behavior).
    """

    def __init__(
        self,
        embed_model: Any,
        breakpoint_threshold_percentile: int,
        min_sentences_per_chunk: int,
        buffer_size: int,
        max_tokens: int,
        tokenizer: TokenizerProtocol,
    ):
        """Initialize TopicChunking.

        Args:
            embed_model: Embedding model to use. If None, uses settings embedder.
            breakpoint_threshold_percentile: Percentile for distance threshold
                (95 = top 5% of distances become breakpoints). Higher = fewer, larger chunks.
            min_sentences_per_chunk: Minimum sentences per chunk.
            buffer_size: Number of surrounding sentences to include in each
                embedding window (matches SemanticSplitterNodeParser.buffer_size).
                1 = [prev, current, next]; 0 = individual sentences only.
            max_tokens: Token cap enforced at the parser layer via
                resplit_oversized_chunks after splitting.
            tokenizer: Tokenizer for the parser-layer resplit guard.
        """
        self._embed_model = embed_model
        self.breakpoint_threshold_percentile = breakpoint_threshold_percentile
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self._buffer_size = buffer_size
        self.max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._splitter = None

    def _get_splitter(self):
        """Lazily return the shared spaCy pipeline (parser-driven doc.sents)."""
        if self._splitter is None:
            self._splitter = get_spacy_nlp()
        return self._splitter

    def _split_into_sentences(self, text: str) -> List[tuple]:
        """Split text into sentences with positions.

        Returns:
            List of (sentence, start_pos, end_pos) tuples.
        """
        nlp = self._get_splitter()
        doc = nlp(text)
        result = []
        for sent in doc.sents:
            txt = sent.text.strip()
            if txt:
                result.append((txt, sent.start_char, sent.end_char))
        return result

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """Chunk text using topic-based boundary detection.

        Args:
            text: Input text to chunk.

        Returns:
            List of Chunk objects.

        Raises:
            ValueError: If embedding fails.
        """
        if not text:
            return []

        sentences_with_pos = self._split_into_sentences(text)
        if not sentences_with_pos:
            return []

        if len(sentences_with_pos) == 1:
            sentence_text = sentences_with_pos[0][0]
            return [Chunk(
                text=sentence_text,
                index=0,
                start_char=sentences_with_pos[0][1],
                end_char=sentences_with_pos[0][2],
            )]

        sentences = [
            _TextWithPos(s[0], s[1], s[2])
            for s in sentences_with_pos
        ]

        ranges = split_by_topic(
            sentences,
            TopicSplitParams(
                embed_model=self._embed_model,
                breakpoint_threshold_percentile=self.breakpoint_threshold_percentile,
                min_sentences_per_chunk=self.min_sentences_per_chunk,
                buffer_size=self._buffer_size,
            ),
        )

        chunks: List[Chunk] = []
        for chunk_idx, (start_i, end_i) in enumerate(ranges):
            chunk_text = " ".join(sentences[i].text for i in range(start_i, end_i + 1))
            start_pos = sentences[start_i].start_char
            end_pos = sentences[end_i].end_char
            chunks.append(Chunk(
                text=chunk_text,
                index=len(chunks),
                start_char=start_pos,
                end_char=end_pos,
                metadata={"topic_boundary": chunk_idx < len(ranges) - 1},
            ))

        if not chunks:
            chunks = [Chunk(
                text=sentences[0].text,
                index=0,
                start_char=sentences[0].start_char,
                end_char=sentences[0].end_char,
            )]

        logger.debug(f"Created {len(chunks)} topic-based chunks")

        if self.max_tokens and self._tokenizer:
            from .constraints import resplit_oversized_chunks
            nlp = self._get_splitter()
            chunks = resplit_oversized_chunks(
                chunks,
                cap_tokens=self.max_tokens,
                tokenizer=self._tokenizer,
                sentence_splitter=lambda t: [s.text for s in nlp(t).sents],
            )
            logger.debug(f"Resplit to {len(chunks)} chunks after token cap enforcement")

        return chunks
