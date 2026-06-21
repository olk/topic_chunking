"""Tests for DoclingTopicChunking and _topic_split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import (
    DocItemLabel,
    ProvenanceItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel as Label

from src._body_resplit import resplit_body
from src._topic_split import TopicSplitParams, split_by_topic
from src.docling_topic import (
    DoclingTopicChunking,
    SentenceWithProv,
    TopicDocMeta,
    _EMPTY_PROV,
)


class _DummyEmbedding:
    dim: int = 128

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vec: np.ndarray | None = None

    def get_text_embedding_batch(self, texts: list[str]) -> list[_DummyEmbedding]:
        return [self._text_to_embedding(t) for t in texts]

    def _text_to_embedding(self, text: str) -> _DummyEmbedding:
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, ch in enumerate(text):
            vec[i % self.dim] += ord(ch)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        out = _DummyEmbedding(self.dim)
        out._vec = vec
        return out

    def cosine_similarity(self, other: _DummyEmbedding) -> float:
        if self._vec is None or other._vec is None:
            return 0.0
        return float(np.dot(self._vec, other._vec))


class _Sent:
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


class TestTopicSplitParams:
    def test_empty_sentences_returns_empty(self):
        params = TopicSplitParams(embed_model=_DummyEmbedding())
        result = split_by_topic([], params)
        assert result == []

    def test_single_sentence_returns_single_range(self):
        params = TopicSplitParams(embed_model=_DummyEmbedding())
        result = split_by_topic([_Sent("hello")], params)
        assert result == [(0, 0)]

    def test_very_similar_sentences_produces_single_chunk(self):
        embed = _DummyEmbedding()
        params = TopicSplitParams(embed_model=embed, breakpoint_threshold_percentile=95)
        sentences = [_Sent("The weather is nice today"), _Sent("The weather is nice tomorrow")]
        result = split_by_topic(sentences, params)
        assert result == [(0, 1)]

    def test_dissimilar_sentences_produces_breakpoint(self):
        embed = _DummyEmbedding()
        params = TopicSplitParams(embed_model=embed, breakpoint_threshold_percentile=50)
        sentences = [
            _Sent("The weather is nice today"),
            _Sent("Python 3.12 introduces new features"),
            _Sent("More sentences about programming"),
        ]
        result = split_by_topic(sentences, params)
        assert len(result) >= 2


class TestSentenceWithProv:
    def test_empty_prov_uses_EMPTY_PROV(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="Hello world.",
            orig="",
            prov=[],
        )
        prov = _EMPTY_PROV
        swp = SentenceWithProv(
            text="Hello world.",
            item=item,
            item_sent_start=0,
            item_sent_end=11,
            prov_start=prov,
            prov_end=prov,
        )
        assert swp.prov_start.page_no == 0
        assert swp.prov_start.charspan == (0, 0)

    def test_adjusted_prov_shifts_charspan(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="Sentence one. Sentence two.",
            orig="",
            prov=[
                ProvenanceItem(
                    page_no=3,
                    bbox=BoundingBox(l=10, t=20, r=100, b=40),
                    charspan=(10, 40),
                )
            ],
        )
        base = item.prov[0]
        adj = ProvenanceItem(
            page_no=base.page_no,
            bbox=base.bbox,
            charspan=(base.charspan[0] + 5, base.charspan[0] + 15),
        )
        assert adj.charspan == (15, 25)
        assert adj.page_no == 3


class TestTopicDocMeta:
    def test_topic_doc_meta_fields(self):
        meta = TopicDocMeta(
            doc_items=[],
            headings=["Section 1"],
            source_charspan=(10, 50),
            source_pages=[1, 2],
            source_bboxes=[{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
        )
        assert meta.source_charspan == (10, 50)
        assert meta.source_pages == [1, 2]
        assert len(meta.source_bboxes) == 1


class TestDoclingTopicChunkingUnit:
    @pytest.fixture(autouse=True)
    def _check_spacy(self):
        try:
            import spacy as _
        except ImportError:
            pytest.skip("spacy not installed")

    def test_sentences_with_prov_parses_sentences(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="Sentence one. Sentence two. Sentence three.",
            orig="",
            prov=[],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        from src._spacy_model import get_spacy_nlp

        nlp = get_spacy_nlp()
        sentences = chunker._sentences_with_prov(item, nlp)
        texts = [s.text for s in sentences]
        assert texts == ["Sentence one.", "Sentence two.", "Sentence three."]

    def test_sentences_with_prov_with_prov_adjusts_charspan(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="First sentence. Second sentence.",
            orig="",
            prov=[
                ProvenanceItem(
                    page_no=2,
                    bbox=BoundingBox(l=5, t=10, r=200, b=30),
                    charspan=(100, 150),
                )
            ],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        from src._spacy_model import get_spacy_nlp

        nlp = get_spacy_nlp()
        sentences = chunker._sentences_with_prov(item, nlp)
        assert len(sentences) == 2
        first = sentences[0]
        assert first.prov_start.page_no == 2
        assert first.prov_start.charspan[0] >= 100
        assert first.prov_start.charspan[1] <= 150

    def test_sentences_with_prov_fallback_text(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="Original text.",
            orig="",
            prov=[],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        from src._spacy_model import get_spacy_nlp

        nlp = get_spacy_nlp()
        sentences = chunker._sentences_with_prov(item, nlp, fallback_text="Fallback sentence one. Fallback sentence two.")
        assert len(sentences) == 2
        assert "Fallback" in sentences[0].text


class TestSentencesFullTextWithProv:
    def test_empty_items_list(self):
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        nlp = _SentenceNLP(["ignore"])
        result = chunker._sentences_full_text_with_prov([], nlp)
        assert result == []

    def test_single_item_single_sentence(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="Hello world.",
            orig="",
            prov=[],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        nlp = _SentenceNLP(["Hello world."])
        result = chunker._sentences_full_text_with_prov([item], nlp)
        assert len(result) == 1
        assert result[0].text == "Hello world."
        assert result[0].item is item
        assert result[0].prov_start == _EMPTY_PROV

    def test_single_item_multiple_sentences(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="First sentence. Second sentence.",
            orig="",
            prov=[],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        nlp = _SentenceNLP(["First sentence.", "Second sentence."])
        result = chunker._sentences_full_text_with_prov([item], nlp)
        assert len(result) == 2
        assert result[0].text == "First sentence."
        assert result[1].text == "Second sentence."
        assert result[0].item is item
        assert result[1].item is item

    def test_two_items_two_sentences(self):
        item1 = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="First item sentence.",
            orig="",
            prov=[],
        )
        item2 = TextItem(
            self_ref="#/t/1",
            label=Label.TEXT,
            text="Second item sentence.",
            orig="",
            prov=[],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        nlp = _SentenceNLP(["First item sentence.", "Second item sentence."])
        result = chunker._sentences_full_text_with_prov([item1, item2], nlp)
        assert len(result) == 2
        assert result[0].text == "First item sentence."
        assert result[0].item is item1
        assert result[1].text == "Second item sentence."
        assert result[1].item is item2

    def test_und_continuation_sentence_not_split(self):
        item1 = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="Das gleiche gilt für alle Umstände, die Ihnen auffallen",
            orig="",
            prov=[],
        )
        item2 = TextItem(
            self_ref="#/t/1",
            label=Label.TEXT,
            text="und die Ihnen nach der Lebenserfahrung ungewöhnlich erscheinen.",
            orig="",
            prov=[],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        full_sent = (
            "Das gleiche gilt für alle Umstände, die Ihnen auffallen "
            "und die Ihnen nach der Lebenserfahrung ungewöhnlich erscheinen."
        )
        nlp = _SentenceNLP([full_sent])
        result = chunker._sentences_full_text_with_prov([item1, item2], nlp)
        assert len(result) == 1, f"Expected 1 sentence (not split on 'und'), got {len(result)}: {[r.text for r in result]}"
        assert "und die Ihnen nach" in result[0].text
        assert result[0].item is item1

    def test_item_with_prov_charspan_adjusted(self):
        item = TextItem(
            self_ref="#/t/0",
            label=Label.TEXT,
            text="Prov sentence one. Prov sentence two.",
            orig="",
            prov=[
                ProvenanceItem(
                    page_no=4,
                    bbox=BoundingBox(l=10, t=20, r=300, b=40),
                    charspan=(50, 120),
                )
            ],
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        nlp = _SentenceNLP(["Prov sentence one.", "Prov sentence two."])
        result = chunker._sentences_full_text_with_prov([item], nlp)
        assert len(result) == 2
        assert result[0].prov_start.page_no == 4
        assert result[0].prov_start.charspan[0] >= 50
        assert result[0].prov_start.charspan[1] <= 120


class TestToProjectChunk:
    def test_maps_charspan_to_start_end(self):
        from docling_core.transforms.chunker.doc_chunk import DocChunk

        doc_chunk = DocChunk(
            text="Test chunk text",
            meta=TopicDocMeta(
                doc_items=[],
                source_charspan=(25, 75),
                source_pages=[3],
                source_bboxes=[{"page_no": 3, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
            ),
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        proj = chunker.to_project_chunk(doc_chunk, 5)
        assert proj.start_char == 25
        assert proj.end_char == 75
        assert proj.metadata["page_no"] == 3
        assert proj.metadata["bboxes"] == [{"page_no": 3, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]

    def test_handles_missing_charspan(self):
        from docling_core.transforms.chunker.doc_chunk import DocChunk

        doc_chunk = DocChunk(
            text="Test chunk text",
            meta=TopicDocMeta(doc_items=[]),
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        proj = chunker.to_project_chunk(doc_chunk, 0)
        assert proj.start_char == 0
        assert proj.end_char == len("Test chunk text")

    def test_handles_empty_source_pages(self):
        from docling_core.transforms.chunker.doc_chunk import DocChunk

        doc_chunk = DocChunk(
            text="Test",
            meta=TopicDocMeta(doc_items=[], source_charspan=(0, 4)),
        )
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed)
        proj = chunker.to_project_chunk(doc_chunk, 0)
        assert proj.metadata["page_no"] is None


class _ApproxTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0] * (len(text) // 4 + 1)

    def decode(self, tokens: list[int]) -> str:
        return "[approx]" * len(tokens)


class _SentSpan:
    __slots__ = ("text", "start_char", "end_char")

    def __init__(self, text: str, start_char: int, end_char: int):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class _SentenceNLP:
    def __init__(self, sentences: list[str]):
        self._sentences = sentences

    def __call__(self, text: str) -> "_SentenceDoc":
        return _SentenceDoc(text, self._sentences)


class _SentenceDoc:
    def __init__(self, text: str, sentences: list[str]):
        self.text = text
        spans: list[_SentSpan] = []
        cursor = 0
        for sent in sentences:
            pos = text.find(sent, cursor)
            if pos == -1:
                pos = cursor
            spans.append(_SentSpan(sent, pos, pos + len(sent)))
            cursor = pos + len(sent)
        self._spans = spans

    @property
    def sents(self):
        return iter(self._spans)


class TestCapEnforcement:
    def test_max_tokens_zero_no_tokenizer_is_valid(self):
        embed = _DummyEmbedding()
        chunker = DoclingTopicChunking(embed_model=embed, max_tokens=0, tokenizer=None)
        assert chunker.max_tokens == 0
        assert chunker.tokenizer is None

    def test_max_tokens_with_tokenizer_is_valid(self):
        embed = _DummyEmbedding()
        tok = _ApproxTokenizer()
        chunker = DoclingTopicChunking(embed_model=embed, max_tokens=100, tokenizer=tok)
        assert chunker.max_tokens == 100

    def test_max_tokens_without_tokenizer_raises(self):
        embed = _DummyEmbedding()
        with pytest.raises(ValueError, match="tokenizer is required"):
            DoclingTopicChunking(embed_model=embed, max_tokens=100, tokenizer=None)

    def test_body_under_cap_yields_one_chunk(self):
        embed = _DummyEmbedding()
        tok = _ApproxTokenizer()
        chunker = DoclingTopicChunking(embed_model=embed, max_tokens=1000, tokenizer=tok)
        heading_prefix = "Title > Section\n"
        heading_tokens = len(tok.encode(heading_prefix))
        body = " ".join(["short sentence."] * 5)
        body_tokens = len(tok.encode(body))
        total = heading_tokens + body_tokens
        assert total < 1000
        sentence_offsets = chunker._build_sentence_offsets([])
        sents = []
        for i, w in enumerate(["short"] * 5):
            swp = SentenceWithProv(
                text=w + ".",
                item=TextItem(self_ref=f"#/t/{i}", label=Label.TEXT, text=w + ".", orig="", prov=[]),
                item_sent_start=0,
                item_sent_end=len(w + "."),
                prov_start=_EMPTY_PROV,
                prov_end=_EMPTY_PROV,
            )
            sents.append(swp)
        sentence_offsets = chunker._build_sentence_offsets(sents)
        found = chunker._find_sentences_in_body_range(sents, 0, len("short. " * 5), sentence_offsets)
        assert len(found) == 5

    def test_heading_exceeds_cap_raises(self):
        embed = _DummyEmbedding()
        tok = _ApproxTokenizer()
        with pytest.raises(ValueError, match="tokenizer is required"):
            DoclingTopicChunking(embed_model=embed, max_tokens=100, tokenizer=None)

    def test_sub_chunks_each_carry_heading(self):
        embed = _DummyEmbedding()
        tok = _ApproxTokenizer()
        chunker = DoclingTopicChunking(embed_model=embed, max_tokens=100, tokenizer=tok)
        heading_prefix = "T > S\n"
        heading_tokens = len(tok.encode(heading_prefix))
        body_cap = 100 - heading_tokens
        sent = "word. "
        body = sent * 200
        nlp = _SentenceNLP([sent] * 200)
        subs = resplit_body(body, body_cap, tok, nlp)
        assert len(subs) > 1
        for sub_text, _, _ in subs:
            assert sub_text in body
            full_text = heading_prefix + sub_text
            assert full_text.startswith(heading_prefix)

    def test_resplit_body_returns_offsets(self):
        tok = _ApproxTokenizer()
        body = "one. two. three. four."
        sents = ["one.", "two.", "three.", "four."]
        nlp = _SentenceNLP(sents)
        subs = resplit_body(body, 2, tok, nlp)
        assert len(subs) > 1
        for sub_text, start, end in subs:
            assert sub_text == body[start:end]
            assert start < end
            assert 0 <= start < len(body)
            assert 0 < end <= len(body)

    def test_resplit_body_empty_string(self):
        tok = _ApproxTokenizer()
        nlp = _SentenceNLP([])
        result = resplit_body("", 10, tok, nlp)
        assert result == []

    def test_resplit_body_cap_too_small_raises(self):
        tok = _ApproxTokenizer()
        nlp = _SentenceNLP(["some text."])
        with pytest.raises(ValueError, match="cap_tokens must be positive"):
            resplit_body("some text", 0, tok, nlp)

    def test_resplit_body_never_cuts_inside_sentence(self):
        tok = _ApproxTokenizer()
        sents = [
            "Short sentence one.",
            "Short sentence two.",
            "word " * 20 + "word.",  # one long oversize sentence
            "Short sentence three.",
        ]
        body = " ".join(sents)
        nlp = _SentenceNLP(sents)
        cap = 6
        subs = resplit_body(body, cap, tok, nlp)
        assert len(subs) >= 2
        for sub_text, start, end in subs:
            assert sub_text == body[start:end]
        long_parts = [s for s in subs if "word word" in s[0]]
        assert len(long_parts) == 1, "Long oversize sentence should be preserved intact"
        assert long_parts[0][0].endswith("word.")

    def test_resplit_body_oversize_single_sentence_emitted_atomically(self):
        tok = _ApproxTokenizer()
        long_sent = "word word word word."
        nlp = _SentenceNLP([long_sent])
        body = long_sent
        cap = 3  # tighter than the sentence's token count
        subs = resplit_body(body, cap, tok, nlp)
        assert len(subs) == 1
        sub_text, start, end = subs[0]
        assert sub_text == long_sent
        assert start == 0
        assert end == len(long_sent)

    def test_find_sentences_in_body_range_partial(self):
        embed = _DummyEmbedding()
        tok = _ApproxTokenizer()
        chunker = DoclingTopicChunking(embed_model=embed, max_tokens=1000, tokenizer=tok)
        s1 = SentenceWithProv(
            text="Apples are red.",
            item=TextItem(self_ref="#/t/0", label=Label.TEXT, text="Apples are red.", orig="", prov=[
                ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, 15))
            ]),
            item_sent_start=0,
            item_sent_end=15,
            prov_start=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, 15)),
            prov_end=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, 15)),
        )
        s2 = SentenceWithProv(
            text="Oranges are orange.",
            item=TextItem(self_ref="#/t/1", label=Label.TEXT, text="Oranges are orange.", orig="", prov=[
                ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(16, 32))
            ]),
            item_sent_start=0,
            item_sent_end=18,
            prov_start=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(16, 32)),
            prov_end=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(16, 32)),
        )
        sentences = [s1, s2]
        offsets = chunker._build_sentence_offsets(sentences)
        assert offsets == [(0, 15), (16, 35)]
        found = chunker._find_sentences_in_body_range(sentences, 16, 35, offsets)
        assert len(found) == 1
        assert found[0].text == "Oranges are orange."

    def test_body_tokens_calculation(self):
        tok = _ApproxTokenizer()
        body = "word " * 10
        tokens = tok.encode(body)
        assert len(tokens) > 0
