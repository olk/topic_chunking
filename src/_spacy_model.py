# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT

"""Shared spaCy de_dep_news_trf pipeline singleton.

All chunking strategies that need sentence segmentation share the same
de_dep_news_trf nlp instance so the BERT weights are loaded only once.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)

_SPACY_MODEL = "de_dep_news_trf"

_DISABLED = ["ner", "tagger", "morphologizer", "lemmatizer", "attribute_ruler"]

_lock = threading.Lock()
_nlp: "spacy.Language | None" = None


def get_spacy_nlp() -> "spacy.Language":
    """Return the cached de_dep_news_trf pipeline (lazy, thread-safe).

    Loads on first call. Subsequent calls return the same instance.
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    with _lock:
        if _nlp is not None:
            return _nlp

        import spacy as _spacy

        try:
            _nlp = _spacy.load(_SPACY_MODEL, disable=_DISABLED)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{_SPACY_MODEL}' not found. "
                f"Install: pip install 'spacy[transformers]>=3.8.14,<3.9.0' && "
                f"python -m spacy download {_SPACY_MODEL}"
            ) from e

        return _nlp


def warmup_spacy_nlp() -> "spacy.Language":
    """Eagerly load the de_dep_news_trf pipeline and run a dummy doc through it.

    This forces the transformer component to initialise its BERT weights so the
    first real chunking request does not pay the ~5-15 s cold-start cost.
    Idempotent — calling multiple times is safe.
    """
    nlp = get_spacy_nlp()
    start = time.monotonic()
    logger.info("Warming up spaCy %s pipeline...", _SPACY_MODEL)
    _ = list(nlp.pipe(["Warming up the spaCy pipeline with a dummy sentence."]))
    elapsed = time.monotonic() - start
    logger.info("spaCy %s ready (warm-up took %.1f s)", _SPACY_MODEL, elapsed)
    return nlp
