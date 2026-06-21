"""Docling-aware topic-based chunker.

Implements :class:`DoclingTopicChunking`, a :class:`BaseChunker` that walks a
:Class:`DoclingDocument` using heading-driven sections, applies the
Greg Kamradt level-3 topic-splitting algorithm inside each section, and yields
:class:`DocChunk` instances whose :class:`TopicDocMeta` carries accurate
source positions (``source_charspan``, ``source_pages``, ``source_bboxes``)
for PDF highlighting.

The public :meth:`DoclingTopicChunking.to_project_chunk` adapter method
converts each :class:`BaseChunk` to the project's :class:`Chunk` model.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import override

from docling_core.transforms.chunker import BaseChunk, BaseChunker
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.serializer.base import BaseSerializerProvider
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    BoundingBox,
    DocItem,
    DocItemLabel,
    DocumentOrigin,
    FloatingItem,
    ListGroup,
    NodeItem,
    OrderedList,
    PictureItem,
    ProvenanceItem,
    SectionHeaderItem,
    TableItem,
    TitleItem,
)

from ._body_resplit import resplit_body
from ._spacy_model import get_spacy_nlp
from ._topic_split import TopicSplitParams, split_by_topic
from .models import Chunk
from .tokenizer import count_tokens

logger = logging.getLogger(__name__)

_EMPTY_BBOX = BoundingBox(l=0.0, t=0.0, r=0.0, b=0.0)
_EMPTY_PROV = ProvenanceItem(page_no=0, bbox=_EMPTY_BBOX, charspan=(0, 0))

_FURNITURE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

_HEADING_SEP = " > "


class ChunkingSerializerProvider(BaseSerializerProvider):
    @override
    def get_serializer(self, doc: DLDocument) -> MarkdownDocSerializer:
        return MarkdownDocSerializer(
            doc=doc,
            params=MarkdownParams(
                image_mode=ImageRefMode.PLACEHOLDER,
                image_placeholder="",
                escape_underscores=False,
                escape_html=False,
            ),
        )


@dataclass(frozen=True)
class SentenceWithProv:
    text: str
    item: DocItem
    item_sent_start: int
    item_sent_end: int
    prov_start: ProvenanceItem
    prov_end: ProvenanceItem


class TopicDocMeta(DocMeta):
    source_charspan: Optional[tuple[int, int]] = Field(default=None)
    source_pages: list[int] = Field(default_factory=list)
    source_bboxes: list[dict] = Field(default_factory=list)

    @field_validator("doc_items", mode="wrap")
    @classmethod
    def _allow_empty_doc_items(cls, v, info):
        return v if v else []


class DoclingTopicChunking(BaseChunker):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embed_model: Any
    breakpoint_threshold_percentile: int = 95
    min_sentences_per_chunk: int = 1
    buffer_size: int = 1
    max_tokens: int = 0
    """Token cap per chunk. If > 0, each emitted chunk's total tokens
    (heading prefix + body) must be ≤ this value. Requires ``tokenizer``
    to be set. Raises ``ValueError`` at chunk time if exceeded."""
    tokenizer: Any = None
    """Tokenizer for cap enforcement. Required when ``max_tokens > 0``."""
    always_emit_headings: bool = False
    serializer_provider: BaseSerializerProvider = field(
        default_factory=ChunkingSerializerProvider
    )

    @model_validator(mode="after")
    def _check_tokenizer_required_when_capping(self) -> "DoclingTopicChunking":
        if self.max_tokens > 0 and self.tokenizer is None:
            raise ValueError(
                "tokenizer is required when max_tokens > 0 "
                f"(got max_tokens={self.max_tokens}, tokenizer=None)"
            )
        return self

    @override
    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        nlp = get_spacy_nlp()
        heading_by_level: dict[int, TitleItem | SectionHeaderItem] = {}
        section_items: list[DocItem] = []

        for item, _lvl in dl_doc.iterate_items(with_groups=True):
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                if section_items:
                    yield from self._emit_section(
                        section_items, heading_by_level, dl_doc, nlp
                    )
                    section_items = []
                level = item.level if isinstance(item, SectionHeaderItem) else 0
                for k in [k for k in heading_by_level if k >= level]:
                    heading_by_level.pop(k, None)
                heading_by_level[level] = item
                continue

            if isinstance(item, (ListGroup, OrderedList, NodeItem)) and not isinstance(
                item, (DocItem, FloatingItem)
            ):
                continue

            if isinstance(item, DocItem) and item.label not in _FURNITURE_LABELS:
                section_items.append(item)

        if section_items:
            yield from self._emit_section(section_items, heading_by_level, dl_doc, nlp)

    def _emit_section(
        self,
        items: list[DocItem],
        heading_by_level: dict[int, TitleItem | SectionHeaderItem],
        dl_doc: DLDocument,
        nlp,
    ) -> Iterator[BaseChunk]:
        text_items: list[DocItem] = []
        other_items: list[tuple[DocItem, str]] = []
        for item in items:
            if isinstance(item, TableItem):
                cap_text = self._serialize_table(item, dl_doc)
                other_items.append((item, cap_text))
            elif isinstance(item, PictureItem):
                cap_text = self._picture_caption_text(item, dl_doc)
                other_items.append((item, cap_text))
            else:
                text_items.append(item)

        sentences: list[SentenceWithProv] = []
        if text_items:
            sentences.extend(self._sentences_full_text_with_prov(text_items, nlp))
        for item, cap_text in other_items:
            sentences.extend(self._sentences_with_prov(item, nlp, fallback_text=cap_text))

        if not sentences:
            return

        ranges = split_by_topic(
            sentences,
            TopicSplitParams(
                embed_model=self.embed_model,
                breakpoint_threshold_percentile=self.breakpoint_threshold_percentile,
                min_sentences_per_chunk=self.min_sentences_per_chunk,
                buffer_size=self.buffer_size,
            ),
        )

        headings = [h.text for h in heading_by_level.values()]
        heading_prefix = (_HEADING_SEP.join(headings) + "\n") if headings else ""

        heading_tokens = 0
        body_cap: int | None = None
        if self.max_tokens > 0 and self.tokenizer is not None:
            heading_tokens = (
                count_tokens(heading_prefix, self.tokenizer) if heading_prefix else 0
            )
            body_cap = self.max_tokens - heading_tokens
            if body_cap <= 0:
                raise ValueError(
                    f"heading tokens ({heading_tokens}) consume entire max_tokens "
                    f"({self.max_tokens}); no room for body"
                )

        sentence_offsets = self._build_sentence_offsets(sentences)

        for start_i, end_i in ranges:
            sents = sentences[start_i : end_i + 1]
            body = " ".join(s.text for s in sents)
            first, last = sents[0], sents[-1]

            body_tokens = (
                count_tokens(body, self.tokenizer) if self.tokenizer else 0
            )

            if body_cap is not None and body_tokens > body_cap:
                sub_bodies = resplit_body(
                    body,
                    body_cap,
                    self.tokenizer,
                    nlp,
                )
                for sub_text, sub_start, sub_end in sub_bodies:
                    sub_sents = self._find_sentences_in_body_range(
                        sentences, sub_start, sub_end, sentence_offsets
                    )
                    sub_first = sub_sents[0] if sub_sents else first
                    sub_last = sub_sents[-1] if sub_sents else last
                    yield DocChunk(
                        text=heading_prefix + sub_text,
                        meta=TopicDocMeta(
                            doc_items=[s.item for s in sub_sents] if sub_sents else [s.item for s in sents],
                            headings=headings or None,
                            origin=dl_doc.origin,
                            source_charspan=(
                                sub_first.prov_start.charspan[0],
                                sub_last.prov_end.charspan[1],
                            ),
                            source_pages=[s.prov_start.page_no for s in sub_sents] if sub_sents else [s.prov_start.page_no for s in sents],
                            source_bboxes=[
                                {
                                    "page_no": s.prov_start.page_no,
                                    "bbox": s.prov_start.bbox.model_dump(),
                                }
                                for s in sub_sents
                            ] if sub_sents else [
                                {
                                    "page_no": s.prov_start.page_no,
                                    "bbox": s.prov_start.bbox.model_dump(),
                                }
                                for s in sents
                            ],
                        ),
                    )
            else:
                yield DocChunk(
                    text=heading_prefix + body,
                    meta=TopicDocMeta(
                        doc_items=[s.item for s in sents],
                        headings=headings or None,
                        origin=dl_doc.origin,
                        source_charspan=(
                            first.prov_start.charspan[0],
                            last.prov_end.charspan[1],
                        ),
                        source_pages=[s.prov_start.page_no for s in sents],
                        source_bboxes=[
                            {
                                "page_no": s.prov_start.page_no,
                                "bbox": s.prov_start.bbox.model_dump(),
                            }
                            for s in sents
                        ],
                    ),
                )

    def _build_sentence_offsets(
        self, sentences: list[SentenceWithProv]
    ) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for s in sentences:
            offsets.append((cursor, cursor + len(s.text)))
            cursor += len(s.text) + 1
        return offsets

    def _find_sentences_in_body_range(
        self,
        sentences: list[SentenceWithProv],
        sub_start: int,
        sub_end: int,
        sentence_offsets: list[tuple[int, int]],
    ) -> list[SentenceWithProv]:
        overlapping: list[SentenceWithProv] = []
        for i, (off_start, off_end) in enumerate(sentence_offsets):
            if off_start < sub_end and off_end > sub_start:
                overlapping.append(sentences[i])
        return overlapping

    def _sentences_with_prov(
        self,
        item: DocItem,
        nlp,
        fallback_text: str | None = None,
    ) -> list[SentenceWithProv]:
        text = fallback_text if fallback_text is not None else getattr(item, "text", "")
        out: list[SentenceWithProv] = []
        if not item.prov:
            doc = nlp(text)
            for sent in doc.sents:
                txt = sent.text.strip()
                if not txt:
                    continue
                out.append(
                    SentenceWithProv(
                        text=txt,
                        item=item,
                        item_sent_start=0,
                        item_sent_end=len(txt),
                        prov_start=_EMPTY_PROV,
                        prov_end=_EMPTY_PROV,
                    )
                )
            return out

        base = item.prov[0]
        doc = nlp(text)
        for sent in doc.sents:
            cs, ce = sent.start_char, sent.end_char
            adj = ProvenanceItem(
                page_no=base.page_no,
                bbox=base.bbox,
                charspan=(base.charspan[0] + cs, base.charspan[0] + ce),
            )
            out.append(
                SentenceWithProv(
                    text=sent.text.strip(),
                    item=item,
                    item_sent_start=cs,
                    item_sent_end=ce,
                    prov_start=adj,
                    prov_end=adj,
                )
            )
        return out

    def _sentences_full_text_with_prov(
        self,
        items: list[DocItem],
        nlp,
    ) -> list[SentenceWithProv]:
        if not items:
            return []

        item_ranges: list[tuple[DocItem, int, int]] = []
        text_parts: list[str] = []
        cursor = 0
        for item in items:
            text = getattr(item, "text", "")
            if not text:
                continue
            item_ranges.append((item, cursor, cursor + len(text)))
            text_parts.append(text)
            cursor += len(text) + 1

        if not text_parts:
            return []

        concatenated = " ".join(text_parts)
        doc = nlp(concatenated)

        out: list[SentenceWithProv] = []
        for sent in doc.sents:
            txt = sent.text.strip()
            if not txt:
                continue
            sent_start = sent.start_char
            sent_end = sent.end_char

            owning_item: DocItem | None = None
            rel_start = 0
            rel_end = 0
            for item, item_start, item_end in item_ranges:
                if item_start <= sent_start < item_end:
                    owning_item = item
                    rel_start = sent_start - item_start
                    rel_end = sent_end - item_start
                    break
                if item_start <= sent_end <= item_end and owning_item is None:
                    owning_item = item
                    rel_start = sent_start - item_start
                    rel_end = sent_end - item_start
                    break

            if owning_item is None:
                continue

            if owning_item.prov:
                base = owning_item.prov[0]
                adj = ProvenanceItem(
                    page_no=base.page_no,
                    bbox=base.bbox,
                    charspan=(base.charspan[0] + rel_start, base.charspan[0] + rel_end),
                )
                out.append(
                    SentenceWithProv(
                        text=txt,
                        item=owning_item,
                        item_sent_start=rel_start,
                        item_sent_end=rel_end,
                        prov_start=adj,
                        prov_end=adj,
                    )
                )
            else:
                out.append(
                    SentenceWithProv(
                        text=txt,
                        item=owning_item,
                        item_sent_start=rel_start,
                        item_sent_end=rel_end,
                        prov_start=_EMPTY_PROV,
                        prov_end=_EMPTY_PROV,
                    )
                )

        return out

    def _picture_caption_text(self, item: PictureItem, dl_doc: DLDocument) -> str:
        return item.caption_text(dl_doc)

    def _serialize_table(self, item: TableItem, dl_doc: DLDocument) -> str:
        df = item.export_to_dataframe(doc=dl_doc)
        if df.empty:
            return ""
        lines = []
        for col in df.columns:
            lines.append(str(col).strip())
        for _, row in df.iterrows():
            row_parts = [str(v).strip() for v in row.values if str(v).strip()]
            if row_parts:
                lines.append(", ".join(row_parts))
        return "; ".join(lines)

    def to_project_chunk(self, doc_chunk: BaseChunk, index: int) -> Chunk:
        meta = doc_chunk.meta
        span = getattr(meta, "source_charspan", None) or (0, len(doc_chunk.text))
        doc_item_refs = []
        doc_items = getattr(meta, "doc_items", None)
        if doc_items:
            for it in doc_items:
                if hasattr(it, "self_ref"):
                    doc_item_refs.append(it.self_ref)
        source_pages = getattr(meta, "source_pages", None)
        return Chunk(
            text=doc_chunk.text,
            index=index,
            start_char=span[0],
            end_char=span[1],
            metadata={
                "headings": getattr(meta, "headings", None),
                "doc_item_refs": doc_item_refs,
                "page_no": source_pages[0] if source_pages else None,
                "bboxes": getattr(meta, "source_bboxes", []),
            },
        )
