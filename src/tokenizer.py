from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    import tiktoken
    import transformers


@runtime_checkable
class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...


class OllamaTokenizer:
    base_url: str

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def encode(self, text: str) -> list[int]:
        response = httpx.post(
            f"{self.base_url}/api/tokenize",
            json={"content": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("tokens", [])

    def decode(self, tokens: list[int]) -> str:
        response = httpx.post(
            f"{self.base_url}/api/detokenize",
            json={"tokens": tokens},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("content", "")


class TiktokenTokenizer:
    def __init__(self, model: str):
        import tiktoken
        self.encoding = tiktoken.encoding_for_model(model)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)


class HuggingFaceTokenizer:
    tokenizer: transformers.AutoTokenizer

    def __init__(self, model: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)  # type: ignore[attr-defined]


class ApproximateTokenizer:
    CHARS_PER_TOKEN = 4

    def encode(self, text: str) -> list[int]:
        return list(range(len(text) // self.CHARS_PER_TOKEN))

    def decode(self, tokens: list[int]) -> str:
        return f"[Approximate: {len(tokens)} tokens]"


@lru_cache(maxsize=4)
def get_tokenizer(
    provider: str,
    base_url: str | None = None,
    model: str | None = None,
    tokenizer_model_id: str | None = None,
) -> TokenizerProtocol:
    """Resolve a tokenizer for the given provider.

    The returned tokenizer implements the ``TokenizerProtocol`` (encode/decode)
    and is used for accurate token counting and truncation.  Pick by provider:

    - ``ollama`` → ``OllamaTokenizer`` (calls the /api/tokenize endpoint)
    - ``openai`` / ``azure`` → ``TiktokenTokenizer`` (uses tiktoken encoding)
    - ``vllm`` / ``hosted_vllm`` / ``tei`` → ``HuggingFaceTokenizer``
      (loads AutoTokenizer from HuggingFace; TEI serves HF models).
      ``tokenizer_model_id`` takes precedence if provided; otherwise falls back
      to ``model``.  On any failure the result is ``ApproximateTokenizer``.
    - default → ``ApproximateTokenizer`` (1 token ≈ 4 chars; only as a
      last-resort fallback)
    """
    provider = provider.lower()
    if provider == "ollama":
        return OllamaTokenizer(base_url or "http://localhost:11434/v1")
    elif provider in ("openai", "azure"):
        return TiktokenTokenizer(model=model or "gpt-4")
    elif provider in ("vllm", "hosted_vllm", "tei"):
        hf_model = tokenizer_model_id or model
        if hf_model:
            try:
                tokenizer = HuggingFaceTokenizer(model=hf_model)
                logging.getLogger(__name__).info(
                    "Loaded HuggingFaceTokenizer(%s) for provider %s", hf_model, provider
                )
                return tokenizer
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to load HuggingFaceTokenizer(%s) for provider %s, "
                    "falling back to ApproximateTokenizer",
                    hf_model, provider,
                )
        return ApproximateTokenizer()
    return ApproximateTokenizer()


def count_tokens(text: str, tokenizer: TokenizerProtocol) -> int:
    return len(tokenizer.encode(text))
