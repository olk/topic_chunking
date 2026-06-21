# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT

"""Chunk data models."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Chunk:
    """A text chunk with metadata.

    Attributes:
        text: The chunk text content.
        index: Zero-based index of the chunk.
        start_char: Character position in source text.
        end_char: End character position in source text.
        metadata: Additional metadata for the chunk.
    """
    text: str
    index: int
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }

    @property
    def char_count(self) -> int:
        """Return the character count of the chunk."""
        return len(self.text)