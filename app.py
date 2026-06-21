"""CLI app: parse a German PDF with docling and chunk it with TopicChunking."""

import logging
import os
import sys
from pathlib import Path

import click

HF_HUB_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or os.environ.get("HF_HOME")
    or (Path.home() / ".cache" / "huggingface" / "hub")
)


def _model_cached_locally(model_name: str) -> bool:
    slug = "models--" + model_name.replace("/", "--")
    return (HF_HUB_CACHE / slug).is_dir()


def _enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_pdf_docling(pdf_path: Path):
    """Convert PDF to DoclingDocument using docling on CPU."""
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.CPU
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(str(pdf_path))
    return result.document


def chunk_text(
    text: str,
    embed_model_name: str,
    breakpoint_percentile: int,
    min_sentences: int,
    buffer_size: int,
    max_tokens: int,
):
    """Chunk text using TopicChunking with HuggingFaceEmbedding."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from src.topic import TopicChunking
    from src.tokenizer import get_tokenizer
    from src.models import Chunk

    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device="cpu",
        cache_folder=str(HF_HUB_CACHE),
        model_kwargs={"local_files_only": True},
    )
    tokenizer = get_tokenizer(provider="vllm", model=embed_model_name)

    chunker = TopicChunking(
        embed_model=embed_model,
        breakpoint_threshold_percentile=breakpoint_percentile,
        min_sentences_per_chunk=min_sentences,
        buffer_size=buffer_size,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )

    return chunker.chunk(text), tokenizer


def chunk_docling_topic(
    dl_doc,
    embed_model_name: str,
    breakpoint_percentile: int,
    min_sentences: int,
    buffer_size: int,
    max_tokens: int,
):
    """Chunk a DoclingDocument using DoclingTopicChunking."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from src.docling_topic import DoclingTopicChunking
    from src.tokenizer import get_tokenizer
    from src.models import Chunk

    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device="cpu",
        cache_folder=str(HF_HUB_CACHE),
        model_kwargs={"local_files_only": True},
    )
    tokenizer = get_tokenizer(provider="vllm", model=embed_model_name)

    chunker = DoclingTopicChunking(
        embed_model=embed_model,
        breakpoint_threshold_percentile=breakpoint_percentile,
        min_sentences_per_chunk=min_sentences,
        buffer_size=buffer_size,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )

    chunks: list[Chunk] = []
    for i, doc_chunk in enumerate(chunker.chunk(dl_doc)):
        chunks.append(chunker.to_project_chunk(doc_chunk, i))

    return chunks, tokenizer


def run(
    pdf_path: Path,
    output_json: Path | None,
    embed_model_name: str,
    breakpoint_percentile: int,
    min_sentences: int,
    buffer_size: int,
    max_tokens: int,
    width: int,
    chunker: str,
) -> None:
    logger = logging.getLogger("app")

    if _model_cached_locally(embed_model_name):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    logger.info("Parsing PDF: %s", pdf_path)
    dl_doc = parse_pdf_docling(pdf_path)
    logger.info("Extracted document (origin=%s)", dl_doc.origin)

    if chunker == "docling-topic":
        logger.info("Starting docling-topic chunking ...")
        chunks, tokenizer = chunk_docling_topic(
            dl_doc,
            embed_model_name=embed_model_name,
            breakpoint_percentile=breakpoint_percentile,
            min_sentences=min_sentences,
            buffer_size=buffer_size,
            max_tokens=max_tokens,
        )
    else:
        text = dl_doc.export_to_markdown()
        logger.info("Extracted %d chars from PDF", len(text))
        if not text.strip():
            logger.error("No text extracted from PDF")
            sys.exit(1)
        logger.info("Starting topic-based chunking ...")
        chunks, tokenizer = chunk_text(
            text,
            embed_model_name=embed_model_name,
            breakpoint_percentile=breakpoint_percentile,
            min_sentences=min_sentences,
            buffer_size=buffer_size,
            max_tokens=max_tokens,
        )

    logger.info("Chunking complete")

    from src.text_format import format_chunks
    from src.tokenizer import count_tokens
    total_tokens = count_tokens(" ".join(c.text for c in chunks), tokenizer)
    click.echo(f"Total: {total_tokens} tokens\n")
    click.echo(format_chunks(chunks, tokenizer, width))

    if output_json:
        import json
        data = [c.to_dict() for c in chunks]
        output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Wrote %d chunks to %s", len(chunks), output_json)


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-json", type=click.Path(path_type=Path), default=None,
              metavar="PATH", help="Write chunk list as JSON to this path.")
@click.option("--embed-model", default="Qwen/Qwen3-Embedding-0.6B",
              help="HuggingFace sentence-transformer model name (default: Qwen/Qwen3-Embedding-0.6B).")
@click.option("--breakpoint-percentile", type=int, default=95,
              help="Percentile for distance threshold — higher = fewer, larger chunks (default: 95).")
@click.option("--min-sentences", type=int, default=1,
              help="Minimum sentences per chunk (default: 1).")
@click.option("--buffer-size", type=int, default=1,
              help="Number of surrounding sentences in each embedding window (default: 1).")
@click.option("-m", "--max-tokens", type=int, default=8000,
              help="Max tokens per chunk; <=0 disables the cap (default: 8000).")
@click.option("-w", "--width", type=int, default=100,
              help="Max chars per line when printing chunks; <=0 disables wrapping (default: 100).")
@click.option("-c", "--chunker", type=click.Choice(["topic", "docling-topic"]), default="topic",
              help="Chunking strategy: 'topic' = text-based (Greg Kamradt), 'docling-topic' = docling-aware with source positions (default: topic).")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(
    pdf_path: Path,
    output_json: Path | None,
    embed_model: str,
    breakpoint_percentile: int,
    min_sentences: int,
    buffer_size: int,
    max_tokens: int,
    width: int,
    chunker: str,
    verbose: bool,
) -> None:
    """Parse a German PDF with docling and chunk it with TopicChunking."""
    setup_logging(verbose)

    run(
        pdf_path=pdf_path,
        output_json=output_json,
        embed_model_name=embed_model,
        breakpoint_percentile=breakpoint_percentile,
        min_sentences=min_sentences,
        buffer_size=buffer_size,
        max_tokens=max_tokens,
        width=width,
        chunker=chunker,
    )


if __name__ == "__main__":
    main()
