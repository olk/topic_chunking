# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT

import textwrap

from .models import Chunk
from .tokenizer import TokenizerProtocol, count_tokens


class _LinePreservingWrapper(textwrap.TextWrapper):
    def wrap(self, text: str) -> list[str]:
        return [
            line
            for para in text.split("\n")
            for line in super().wrap(para) or [""]
        ]


def format_chunk(chunk: Chunk, tokenizer: TokenizerProtocol, width: int) -> str:
    n = count_tokens(chunk.text, tokenizer)
    header = f"chunk[{chunk.index}]: {n} tokens"
    if width <= 0:
        body = chunk.text
    else:
        wrapper = _LinePreservingWrapper(
            width=width, break_long_words=True, replace_whitespace=False
        )
        body = wrapper.fill(chunk.text)
    return f"{header}\n{body}"


def format_chunks(chunks: list[Chunk], tokenizer: TokenizerProtocol, width: int) -> str:
    return "\n\n".join(format_chunk(c, tokenizer, width) for c in chunks)
