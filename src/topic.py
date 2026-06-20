"""Topic-based chunking strategy.

Implements Greg Kamradt's "level-3 semantic chunking" algorithm
(https://youtu.be/8OJC21T2SL4?t=1933). Sentence splitting uses spaCy's
``de_dep_news_trf`` transformer model; sentence boundaries are driven by
the dependency parser (doc.sents). The shared pipeline is loaded once via
:func:`._spacy_model.get_spacy_nlp`.
"""

import logging
from typing import Any, List

import numpy as np

from .tokenizer import TokenizerProtocol
from ._spacy_model import get_spacy_nlp
from .models import Chunk

logger = logging.getLogger(__name__)


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

    def _build_sentence_groups(
        self, sentence_texts: List[str]
    ) -> List[str]:
        """Build sliding-window combined sentences for embedding.

        For each index i, the combined sentence is
        ``texts[i-buffer_size] + ... + texts[i] + ... + texts[i+buffer_size]``,
        clipped at document boundaries.  Mirrors LlamaIndex's
        ``SemanticSplitterNodeParser._build_sentence_groups``.

        Args:
            sentence_texts: Flat list of individual sentence strings.

        Returns:
            List of combined sentence strings, one per input sentence.
        """
        n = len(sentence_texts)
        groups: List[str] = []
        for i in range(n):
            combined: str = ""
            for j in range(i - self._buffer_size, i):
                if j >= 0:
                    combined += sentence_texts[j]
            combined += sentence_texts[i]
            for j in range(i + 1, i + 1 + self._buffer_size):
                if j < n:
                    combined += sentence_texts[j]
            groups.append(combined)
        return groups

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

    def _calculate_distances(
        self, embeddings: List[List[float]], embed_model: Any
    ) -> List[float]:
        """Cosine distance between consecutive embeddings.

        Delegates to ``embed_model.similarity`` (default ``SimilarityMode.DEFAULT = cosine``)
        and converts similarity to distance via ``1 - sim``.
        """
        distances: List[float] = []
        for i in range(len(embeddings) - 1):
            similarity = embed_model.similarity(embeddings[i], embeddings[i + 1])
            distances.append(1.0 - float(similarity))
        return distances

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

        sentence_texts = [s[0] for s in sentences_with_pos]
        windowed_texts = self._build_sentence_groups(sentence_texts)

        try:
            embeddings = self._embed_model.get_text_embedding_batch(windowed_texts)
            distances = self._calculate_distances(embeddings, self._embed_model)
        except Exception as e:
            raise ValueError(f"TopicChunking embedding/similarity failed: {e}") from e

        if len(embeddings) != len(sentence_texts):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(sentence_texts)}"
            )

        if not distances:
            return [Chunk(
                text=text,
                index=0,
                start_char=0,
                end_char=len(text),
            )]

        threshold = self._compute_distance_threshold(distances)

        breakpoints = [0]
        for i, dist in enumerate(distances):
            if dist > threshold:
                breakpoints.append(i + 1)

        chunks = []
        for idx, start_idx in enumerate(breakpoints):
            end_idx = breakpoints[idx + 1] if idx + 1 < len(breakpoints) else len(sentences_with_pos)

            if end_idx - start_idx < self.min_sentences_per_chunk and len(breakpoints) > 1:
                continue

            chunk_text = " ".join(sentences_with_pos[i][0] for i in range(start_idx, end_idx))
            start_pos = sentences_with_pos[start_idx][1]
            end_pos = sentences_with_pos[end_idx - 1][2]

            chunks.append(Chunk(
                text=chunk_text,
                index=len(chunks),
                start_char=start_pos,
                end_char=end_pos,
                metadata={"topic_boundary": idx < len(breakpoints) - 1},
            ))

        if not chunks:
            return [Chunk(
                text=sentence_texts[0],
                index=0,
                start_char=sentences_with_pos[0][1],
                end_char=sentences_with_pos[0][2],
            )]

        logger.debug(f"Created {len(chunks)} topic-based chunks")

        if self.max_tokens and self._tokenizer:
            from .constraints import resplit_oversized_chunks
            nlp = self._get_splitter()
            chunks = resplit_oversized_chunks(
                chunks,
                cap_tokens=self.max_tokens,
                tokenizer=self._tokenizer,
                sentence_splitter=lambda text: [s.text for s in nlp(text).sents],
            )
            logger.debug(f"Resplit to {len(chunks)} chunks after token cap enforcement")

        return chunks

    def _compute_distance_threshold(self, distances: List[float]) -> float:
        """Distance threshold = np.percentile(distances, breakpoint_threshold_percentile).

        Any pair whose distance exceeds this threshold becomes a topic boundary.
        With the default 95, the top ~5% of consecutive-pair distances split.
        """
        if not distances:
            return 0.0
        return float(np.percentile(distances, self.breakpoint_threshold_percentile))
