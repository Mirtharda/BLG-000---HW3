"""Text chunking module.

Strategy: fixed-size character chunks with overlap, with soft sentence-
boundary snapping so chunks don't cut mid-sentence.

Design rationale
----------------
* Chunk size 500 chars / overlap 100 chars balances context richness
  against the embedding model's sequence length limit (~256 tokens for
  all-MiniLM-L6-v2).
* Sentence-boundary snapping (±50 chars look-ahead/behind) keeps each
  chunk semantically coherent without requiring an NLP library.
* Overlap ensures that facts near a chunk boundary are covered by at
  least two chunks, reducing retrieval gaps.
* Documents can be arbitrarily large; the sliding-window loop handles
  that without loading everything into memory at once.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CHUNK_SIZE = 500    # target characters per chunk
OVERLAP    = 100    # characters of overlap between consecutive chunks
MIN_CHUNK  = 80     # discard tiny trailing chunks


@dataclass
class Chunk:
    text:        str
    entity:      str
    entity_type: str
    chunk_index: int
    source_url:  str = ""


def chunk_text(
    text: str,
    entity: str,
    entity_type: str,
    source_url: str = "",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    """Split *text* into overlapping character-level chunks.

    The function walks the text with a sliding window.  At each window
    boundary it tries to snap to the nearest sentence-ending punctuation
    within a ±50-character search window so that chunks don't split words
    or sentences abruptly.
    """
    # Normalise whitespace: collapse 3+ newlines, compress spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx   = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Soft snap: look for a sentence boundary near our target end
        if end < len(text):
            window_start = max(0, end - 50)
            window_end   = min(len(text), end + 50)
            region       = text[window_start:window_end]
            # Find the last sentence-ending punctuation followed by whitespace
            last_match = None
            for m in re.finditer(r"[.!?]\s", region):
                last_match = m
            if last_match:
                end = window_start + last_match.end()

        chunk_str = text[start:end].strip()
        if len(chunk_str) >= MIN_CHUNK:
            chunks.append(
                Chunk(
                    text=chunk_str,
                    entity=entity,
                    entity_type=entity_type,
                    chunk_index=idx,
                    source_url=source_url,
                )
            )
            idx += 1

        # Advance start, backing off by overlap so successive chunks share context
        next_start = end - overlap
        if next_start <= start:          # safety: always make progress
            next_start = start + 1
        start = next_start

    return chunks
