# Topic-Based Chunking for German PDFs

This project provides two chunking strategies for German PDFs using docling for PDF parsing and a topic-based semantic chunking algorithm derived from Greg Kamradt's [5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5) (level 3 — semantic chunking).

| Chunker | Input | Provenance | Sentence segmentation |
|---|---|---|---|
| `TopicChunking` | Plain text | None | spaCy on full text |
| `DoclingTopicChunking` | `DoclingDocument` | Page, bbox, charspan, doc_items | spaCy on full text (concatenated from DocItems) |

Both chunkers share the same core algorithm: spaCy sentence segmentation, embedding-based topic boundary detection, greedy sentence joining, and an optional token-cap safety net. They differ in how text enters the pipeline and what metadata is preserved.


---

## The Algorithm: Level-3 Semantic Chunking

Both chunkers implement the same four-step pipeline:

```
1. Sentence segmentation  (spaCy de_dep_news_trf, parser-driven doc.sents)
2. Embedding windowing   (buffer_size surrounding sentences per embedding)
3. Cosine distance      (1 - cosine_similarity between consecutive windows)
4. Percentile threshold (breakpoint_threshold_percentile-th percentile of distances)
```

### Step 1 — Sentence segmentation

spaCy loads the `de_dep_news_trf` transformer pipeline once via `get_spacy_nlp()`. The dependency parser drives `doc.sents`; sentence boundaries are predicted from the parse tree rather than from rules or regex. This correctly handles German comma-separated subordinate clauses, abbreviations (U.S., GmbH), and quoted speech.

**TopicChunking**: calls `nlp(full_text).sents` once on the entire document text (`src/topic.py:93`). The parser sees full context, which is important for long German run-on sentences.

**DoclingTopicChunking**: concatenates all text DocItems in a section with a single space separator, then calls `nlp(concatenated_text).sents` once (`src/docling_topic.py:388–389`). This was changed from per-DocItem calls specifically to avoid mis-segmentation: when a long German sentence spans multiple DocItems (e.g., a paragraph break inside a comma-separated list), per-DocItem calls give the parser no context and it splits at the conjunction "und", "oder", etc. The full-text call gives the parser the surrounding context and it correctly keeps the sentence intact. Each resulting sentence is mapped back to its source DocItem via recorded char offsets (`src/docling_topic.py:391–424`).

### Step 2 — Embedding windowing

Each sentence is embedded as a sliding window of its `buffer_size` neighbors:

```
window(i) = [sentence(i-buffer_size), ..., sentence(i-1), sentence(i), sentence(i+1), ..., sentence(i+buffer_size)]
```

`buffer_size=1` means each sentence is embedded together with one predecessor and one successor. This smooths out noise so single anomalous sentences do not create spurious breakpoints. `buffer_size=0` disables the window and embeds each sentence in isolation.

Implementation: `src/_topic_split.py:92–107` (`_build_sentence_groups`).

### Step 3 — Cosine distance

For each consecutive pair of windows, cosine similarity is computed between their embedding vectors. The distance is `1 - similarity`. This captures how much the semantic focus shifts between consecutive sentence groups.

Implementation: `src/_topic_split.py:110–121` (`_cosine_distances`).

### Step 4 — Percentile breakpoint threshold

All distances are collected and the `breakpoint_threshold_percentile`-th percentile is used as the threshold. Any consecutive pair whose distance exceeds the threshold becomes a chunk boundary.

```
dist(i) > percentile(distances, breakpoint_threshold_percentile)  →  breakpoint after sentence i
```

With the default `percentile=95`, only the top 5% of semantic jumps become boundaries. A higher value produces fewer, larger chunks; a lower value produces more, smaller chunks.

Implementation: `src/_topic_split.py:66–73`.

The `min_sentences_per_chunk` parameter prevents creating trivially small chunks: if a breakpoint would produce fewer than `min_sentences_per_chunk` sentences in the next chunk, the breakpoint is skipped (unless it is the only way to make progress).

Implementation: `src/_topic_split.py:78–83`.

---

## TopicChunking (text-based)

**File**: `src/topic.py`

### Algorithm

```
chunk(text)
  1. sentences = _split_into_sentences(text)          # nlp(full_text).sents, once
  2. if single sentence → return [Chunk(text)]
  3. ranges = split_by_topic(sentences)              # level-3 semantic split
  4. for each (start_i, end_i) in ranges:
       chunk_text = " ".join(s[i].text for i in start_i..end_i)
       emit Chunk(text=chunk_text, start_char=s[start_i].start_char,
                  end_char=s[end_i].end_char)
  5. if max_tokens > 0:
       chunks = resplit_oversized_chunks(chunks, cap=max_tokens,
                                        sentence_splitter=lambda t: [s.text for s in nlp(t).sents])
```

Key implementation detail: `_split_into_sentences` calls `nlp(text).sents` on the **full** text (`src/topic.py:93`). The result is a list of `(sentence_text, start_char, end_char)` tuples.

### Token-cap enforcement

After initial chunking, `resplit_oversized_chunks` (`src/constraints.py:34–211`) re-splits any chunk whose token count exceeds `max_tokens`. The resplit strategy (in priority order):

1. Split on `\n` (line boundaries — preserves document structure)
2. If a line still exceeds the cap, use the `sentence_splitter` (spaCy `nlp(t).sents`)
3. If a sentence still exceeds the cap, fall back to token-level slicing (`tokenizer.encode/decode`)

The `sentence_splitter` is passed as a lambda: `lambda t: [s.text for s in nlp(t).sents]`. This ensures spaCy is used when available and falls back to raw token slicing only as a last resort.

### Strengths

- **Pure-text input**: works with any string, no document model required
- **Single spaCy call**: parser sees full context, long German run-on sentences stay intact
- **Always whole sentences**: chunks are always built from whole spaCy sentences joined with `" "`
- **No dependencies on docling types**: the chunker itself only depends on spaCy and an embedder
- **Fast resplit**: line-level and whitespace-level splitting is cheap

### Weaknesses

- **No provenance**: no page number, no bounding box, no character span for PDF highlighting
- **Heading as body text**: section headers are part of the body text, not separated as metadata
- **Requires caller to extract text**: the caller must already have called `dl_doc.export_to_markdown()` or similar to get a plain text string
- **Parser-based sentence segmentation**: spaCy's dependency parser occasionally mis-splits very long German sentences at coordinating conjunctions ("und", "oder", "aber"). This is inherited by both chunkers.


---

## DoclingTopicChunking (docling-aware)

**File**: `src/docling_topic.py`

### Algorithm

```
chunk(dl_doc)
  1. Walk dl_doc.iterate_items(with_groups=True)
  2. Accumulate DocItems under each heading section
  3. _emit_section(items, heading_prefix)
```

```
_emit_section(items, heading_prefix)
  1. Separate items into text_items and other_items (tables/pictures)
  2. sentences = _sentences_full_text_with_prov(text_items, nlp)   # key difference
  3. For each table/picture item:
       sentences.extend(_sentences_with_prov(item, nlp, fallback_text=generated_text))
  4. ranges = split_by_topic(sentences)
  5. For each (start_i, end_i) in ranges:
       body = " ".join(sentences[j].text for j in start_i..end_i)
       heading_tokens = count_tokens(heading_prefix)
       body_cap = max_tokens - heading_tokens
       if body_tokens > body_cap:
           sub_bodies = resplit_body(body, body_cap, tokenizer, nlp)
           for each (sub_text, sub_start, sub_end) in sub_bodies:
               emit DocChunk(text=heading_prefix + sub_text,
                             meta=TopicDocMeta(doc_items=[...],
                                              source_charspan=(first.prov_start.charspan[0],
                                                              last.prov_end.charspan[1]),
                                              source_pages=[...], source_bboxes=[...]))
       else:
           emit DocChunk(text=heading_prefix + body, meta=...)
```

### The full-text concatenation trick

`_sentences_full_text_with_prov` (`src/docling_topic.py:353–424`) is the key difference from a naive per-DocItem approach:

```python
# Concatenate all text items with a single space separator
item_ranges = []  # (item, start_char, end_char) in the concatenated string
text_parts  = []
cursor = 0
for item in items:
    text = item.text
    item_ranges.append((item, cursor, cursor + len(text)))
    text_parts.append(text)
    cursor += len(text) + 1

concatenated = " ".join(text_parts)   # single space, spaCy normalizes whitespace
doc = nlp(concatenated)               # ONE spaCy call for the whole section
```

Each spaCy sentence is then mapped back to its source DocItem by binary-searching `item_ranges` for the sentence's `start_char`. The sentence's position inside the item is `rel_start = sent_start - item_start`. Provenance is adjusted accordingly:

```python
base = item.prov[0]
adj = ProvenanceItem(
    page_no=base.page_no,
    bbox=base.bbox,
    charspan=(base.charspan[0] + rel_start, base.charspan[0] + rel_end),
)
```

**Why this matters**: a German PDF paragraph may be split into multiple DocItems by docling's text extraction (e.g., at a line break or list-item boundary). Calling spaCy per fragment gives the parser no context, and it mis-segments long German run-on sentences at coordinating conjunctions. Calling spaCy on the concatenated full text gives the parser surrounding context and it correctly identifies comma-separated clauses as one sentence. This is the same approach `TopicChunking` uses, applied per section.

Tables and pictures are handled separately: their text is generated via `_serialize_table` or `_picture_caption_text` and passed to `_sentences_with_prov` with a `fallback_text`. These are appended at the end of the section's sentence list (not interleaved in original document order — a known limitation).

### Sentence-aware body resplit

When a chunk's body exceeds `body_cap` (= `max_tokens - heading_tokens`), `resplit_body` (`src/_body_resplit.py:27–111`) re-splits it. The strategy:

1. Call `nlp(body).sents` to re-segment the body into spaCy sentences
2. Greedily pack sentences into sub-bodies until adding the next sentence would exceed `cap_tokens`
3. **Atomic emission for oversize sentences**: if a single sentence itself exceeds `cap_tokens`, it is emitted intact as its own sub-body and a warning is logged. The cap is best-effort; no mid-sentence slicing ever occurs.

```python
for sent_text, sent_start, sent_end in sentence_units:
    sent_tokens = len(tokenizer.encode(sent_text))
    if sent_tokens > cap_tokens:
        _flush()                    # emit current buffer
        logger.warning("Sentence (%d tokens) exceeds cap (%d); "
                      "emitting atomic without splitting", sent_tokens, cap_tokens)
        out.append((sent_text, sent_start, sent_end))   # whole sentence
        continue
    if buffer_tokens + sent_tokens > cap_tokens:
        _flush()                    # start new sub-body
    buffer_texts.append(sent_text)
    buffer_tokens += sent_tokens
```

### Strengths

- **Accurate provenance**: `TopicDocMeta` carries `source_charspan`, `source_pages`, `source_bboxes`, and `doc_items` for precise PDF highlighting
- **Heading hierarchy preserved**: headings are extracted as a prefix string (`heading > section > subsection`) and prepended to every chunk, not mixed into body text
- **Never mid-sentence cuts**: `resplit_body` is sentence-first; oversize sentences are emitted atomically with a warning, never sliced mid-sentence
- **Handles tables and pictures**: tables are serialized to text and pictures get their caption text; both go through sentence segmentation
- **Full-text spaCy**: concatenation ensures spaCy sees full section context, fixing the cross-DocItem German sentence split bug that per-item calls exhibit

### Weaknesses

- **Requires DoclingDocument**: cannot process plain text; needs the docling document model and the `prov` API
- **Cross-item sentence provenance is first-item-only**: if one spaCy sentence spans N DocItems, all provenance points to the first item. If the sentence crosses a page break, the trailing pages are not represented in `source_pages`
- **Tables/pictures at section end**: table and picture sentences are appended at the end of the section's sentence list, not interleaved in their original document position
- **Higher memory per section**: the full concatenated section text is held in memory for the single spaCy call; large sections may increase memory usage
- **German-only sentence segmentation**: spaCy model is `de_dep_news_trf`; English or multilingual documents are not supported


---

## Side-by-Side Example

### Input text

```markdown
# Wetter und Küche

Heute scheint die Sonne in Berlin. Die Temperatur liegt bei 22 Grad.
Am Nachmittag ziehen Wolken auf. Regen ist möglich.

## Ein Risotto-Rezept

Für ein gutes Risotto braucht man Arborio-Reis, Brühe und Geduld.
Die Zwiebeln werden in Butter glasig gedünstet.
Der Reis wird mit Wein abgelöscht. Die Brühe wird nach und nach zugegeben.
Am Ende rührt man den Parmesan unter.
```

In the `DoclingDocument` model this becomes:
- TitleItem: "Wetter und Küche"
- SectionHeaderItem: "Ein Risotto-Rezept" (level 2)
- TextItem: "Heute scheint die Sonne in Berlin. Die Temperatur liegt bei 22 Grad. Am Nachmittag ziehen Wolken auf. Regen ist möglich."
- TextItem: "Für ein gutes Risotto braucht man Arborio-Reis, Brühe und Geduld. Die Zwiebeln werden in Butter glasig gedünstet. Der Reis wird mit Wein abgelöscht. Die Brühe wird nach und nach zugegeben. Am Ende rührt man den Parmesan unter."

### TopicChunking output

The markdown text is passed to `TopicChunking.chunk()`. spaCy sees the full text and segments it into ~9 sentences. The topic split detects a large semantic jump between "Regen ist möglich." and "Für ein gutes Risotto..." (weather → cooking), creating two ranges.

```
Chunk 0: "Heute scheint die Sonne in Berlin. Die Temperatur liegt bei 22 Grad.
         Am Nachmittag ziehen Wolken auf. Regen ist möglich."
         tokens: 22

Chunk 1: "Für ein gutes Risotto braucht man Arborio-Reis, Brühe und Geduld.
         Die Zwiebeln werden in Butter glasig gedünstet. Der Reis wird mit Wein
         abgelöscht. Die Brühe wird nach und nach zugegeben. Am Ende rührt man
         den Parmesan unter."
         tokens: 46
```

Each chunk is a plain `Chunk` object. No provenance, no heading prefix, no metadata beyond `start_char`/`end_char`.

### DoclingTopicChunking output

`DoclingTopicChunking` walks the document and builds two sections:

**Section 1** — heading: "Wetter und Küche"

```
Chunk 0: "Wetter und Küche\nHeute scheint die Sonne in Berlin. Die Temperatur
         liegt bei 22 Grad. Am Nachmittag ziehen Wolken auf. Regen ist möglich."
         tokens: 24
         meta.headings:       ["Wetter und Küche"]
         meta.source_pages:   [0]
         meta.doc_items:      [text_item_1]
```

**Section 2** — heading: "Wetter und Küche > Ein Risotto-Rezept"

```
Chunk 1: "Wetter und Küche > Ein Risotto-Rezept\nFür ein gutes Risotto braucht
         man Arborio-Reis, Brühe und Geduld. Die Zwiebeln werden in Butter glasig
         gedünstet. Der Reis wird mit Wein abgelöscht. Die Brühe wird nach und
         nach zugegeben. Am Ende rührt man den Parmesan unter."
         tokens: 50
         meta.headings:       ["Wetter und Küche", "Ein Risotto-Rezept"]
         meta.source_pages:   [0]
         meta.doc_items:      [text_item_2]
```

The heading prefix is separate from the body. The provenance (`source_pages`, `doc_items`) points directly to the source DocItem.

### Token-cap example (max_tokens=30)

With `max_tokens=30` and `heading_tokens≈4`, `body_cap=26`. DoclingTopicChunking processes the cooking section's 46-token body:

1. `_sentences_full_text_with_prov` returns 6 spaCy sentences from the concatenated text
2. `split_by_topic` keeps them together (all cooking-domain sentences, low cosine distance)
3. Body tokens (50) > `body_cap` (26), so `resplit_body` is called
4. `resplit_body` packs sentences greedily:

```
sub-body 0: "Für ein gutes Risotto braucht man Arborio-Reis, Brühe und Geduld."
            10 tokens → fits in cap

sub-body 1: "Die Zwiebeln werden in Butter glasig gedünstet. Der Reis wird mit
            Wein abgelöscht. Die Brühe wird nach und nach zugegeben."
            22 tokens → fits in cap

sub-body 2: "Am Ende rührt man den Parmesan unter."
            7 tokens → fits in cap
```

Each sub-body is emitted as a separate `DocChunk` with the same heading prefix prepended. No sentence was cut in half — `resplit_body` only ever packs complete spaCy sentences.


---

## Configuration

| Parameter | Default (CLI) | Type | Effect |
|---|---|---|---|
| `breakpoint_threshold_percentile` | 95 | int | Distance percentile used as breakpoint threshold. Higher → fewer, larger chunks. Range: 1–99. |
| `min_sentences_per_chunk` | 1 | int | Minimum sentences per chunk. Prevents trivially small chunks at the cost of larger ones nearby. |
| `buffer_size` | 1 | int | Number of neighboring sentences in each embedding window. 0 = no window (legacy). 1 = predecessor + successor. |
| `max_tokens` | 8000 | int | Hard token cap enforced per chunk (heading + body). A sentence that itself exceeds this is emitted atomically. 0 = disabled. |
| `always_emit_headings` | False | bool | (DoclingTopicChunking only) Prepend heading prefix even when the section body is empty. |

For the embedding model, the default is `Qwen/Qwen3-Embedding-0.6B`. Any sentence-transformer model that has a `get_text_embedding_batch` method and a `cosine_similarity` method on embedding vectors is compatible.


---

## How to Run

### Setup

```bash
# Install dependencies
uv sync

# Download spaCy German model
uv run python -m spacy download de_dep_news_trf

# Download HuggingFace embedding model (Qwen3-Embedding-0.6B)
# The app downloads it on first run if not cached; set HF_HUB_CACHE for offline use.
```

### CLI

```bash
# TopicChunking (plain text via docling markdown export)
uv run python app.py Merkblatt_f_r_Auslandsreisen.pdf -c topic -m 8000

# DoclingTopicChunking (docling-aware, with provenance)
uv run python app.py Merkblatt_f_r_Auslandsreisen.pdf -c docling-topic -m 8000

# Save chunks as JSON for inspection
uv run python app.py Merkblatt_f_r_Auslandsreisen.pdf -c docling-topic -m 8000 \
    --output-json chunks.json

# Smaller chunks (more breakpoints)
uv run python app.py Merkblatt_f_r_Auslandsreisen.pdf -c docling-topic \
    --breakpoint-percentile 80 -m 256

# Custom embedding model
uv run python app.py doc.pdf -c docling-topic -m 8000 \
    --embed-model BAAI/bge-m3
```


---

## Files

| Concern | File | Purpose |
|---|---|---|
| Sentence segmentation | `src/_spacy_model.py` | Shared `de_dep_news_trf` spaCy pipeline singleton (`get_spacy_nlp`) |
| Topic split algorithm | `src/_topic_split.py` | `split_by_topic`, `_build_sentence_groups`, `_cosine_distances` |
| Text-based chunker | `src/topic.py` | `TopicChunking` class, `_split_into_sentences`, `chunk()` |
| Docling-aware chunker | `src/docling_topic.py` | `DoclingTopicChunking`, `SentenceWithProv`, `_sentences_full_text_with_prov`, `_emit_section` |
| Body resplit (Docling path) | `src/_body_resplit.py` | `resplit_body` — sentence-first, atomic for oversize sentences |
| Chunk resplit (text path) | `src/constraints.py` | `resplit_oversized_chunks` — line → sentence → token-slicing fallback |
| CLI | `app.py` | `parse_pdf_docling`, `chunk_text`, `chunk_docling_topic`, `run` |
| Tests | `tests/test_docling_topic.py` | Unit tests for `DoclingTopicChunking`, `resplit_body`, `split_by_topic` |


---

## Notes and Limitations

- **German only**: the spaCy model is `de_dep_news_trf`. English or multilingual documents are not supported.
- **Embedding model required**: both chunkers require a HuggingFace sentence-transformer model loaded via `llama_index.embeddings.huggingface`. The default is `Qwen/Qwen3-Embedding-0.6B`. Any compatible model works.
- **Parser-based sentence segmentation**: spaCy's dependency parser occasionally mis-splits very long German sentences at coordinating conjunctions. `DoclingTopicChunking` was specifically hardened against cross-DocItem splits (the most common failure mode in PDF processing) by switching to full-section concatenation. A single spaCy sentence that is longer than `max_tokens` is emitted atomically with a warning — no slicing occurs.
- **Token cap is best-effort**: a sentence longer than `max_tokens` is emitted intact and a `logger.warning` is emitted. The chunk will exceed the cap. Increase `max_tokens` or lower the embedding model's context window if this is a concern.
- **Cross-item provenance is first-item-only**: a spaCy sentence that spans multiple DocItems is mapped to the first item for provenance purposes. If the sentence crosses a page break, `source_pages` will only contain the first page.
- **Tables and pictures at section end**: in `DoclingTopicChunking._emit_section`, table and picture sentences are appended after all text-item sentences, not interleaved in their original document order. For most PDFs (where tables and pictures are inlined in running text) this produces incorrect ordering. A future improvement would track per-item positions and interleave sentences accordingly.
