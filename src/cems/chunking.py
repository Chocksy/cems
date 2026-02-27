"""QMD-style chunking for CEMS documents.

Chunks documents into smaller pieces suitable for embedding models.
Default: 800 tokens per chunk with 15% overlap.
Fallback: 3200 characters per chunk with 480 character overlap.

This approach:
- Avoids truncation issues with limited-context embedding models
- Provides better search recall by returning relevant snippets
- Matches QMD's proven chunking strategy
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default chunking parameters (QMD-style)
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_PERCENT = 0.15  # 15% overlap = 120 tokens
DEFAULT_CHUNK_CHARS = 3200  # ~4 chars per token fallback
DEFAULT_OVERLAP_CHARS = 480  # 15% of 3200


@dataclass
class Chunk:
    """A chunk of a document."""

    seq: int  # Sequence number (0-indexed)
    pos: int  # Character position in original document
    content: str  # Chunk text
    tokens: int | None  # Token count (if available)
    bytes: int  # Byte size


def content_hash(content: str) -> str:
    """Generate a SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Chunker:
    """Document chunker with token-based splitting and character fallback.

    Usage:
        chunker = Chunker()
        chunks = chunker.chunk(long_document)
        for chunk in chunks:
            embed(chunk.content)
    """

    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_percent: float = DEFAULT_OVERLAP_PERCENT,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        encoding_name: str = "cl100k_base",  # GPT-4/embedding model tokenizer
    ):
        """Initialize the chunker.

        Args:
            chunk_tokens: Target tokens per chunk
            overlap_percent: Overlap as fraction of chunk size
            chunk_chars: Fallback character limit per chunk
            overlap_chars: Fallback character overlap
            encoding_name: Tiktoken encoding name
        """
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = int(chunk_tokens * overlap_percent)
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.encoding_name = encoding_name
        # None = not yet loaded, False = load failed, otherwise tiktoken Encoding
        self._encoding: "Any" = None

    def _get_encoding(self) -> "Any":
        """Lazy-load tiktoken encoding. Returns Encoding or None on failure."""
        if self._encoding is None:
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding(self.encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")
                self._encoding = False  # Mark as failed
        return self._encoding if self._encoding else None

    def chunk(self, content: str) -> list[Chunk]:
        """Chunk content into smaller pieces.

        Tries token-based chunking first, falls back to character-based.

        Args:
            content: The document content to chunk

        Returns:
            List of Chunk objects
        """
        if not content or not content.strip():
            return []

        # Try token-based chunking
        encoding = self._get_encoding()
        if encoding:
            return self._chunk_by_tokens(content, encoding)

        # Fallback to character-based chunking
        logger.debug("Using character-based chunking (tiktoken unavailable)")
        return self._chunk_by_chars(content)

    def _chunk_by_tokens(self, content: str, encoding) -> list[Chunk]:
        """Chunk content by token count with overlap."""
        tokens = encoding.encode(content)
        total_tokens = len(tokens)

        # If content fits in a single chunk, return as-is
        if total_tokens <= self.chunk_tokens:
            return [
                Chunk(
                    seq=0,
                    pos=0,
                    content=content,
                    tokens=total_tokens,
                    bytes=len(content.encode("utf-8")),
                )
            ]

        chunks = []
        step = self.chunk_tokens - self.overlap_tokens
        seq = 0
        start_token = 0

        while start_token < total_tokens:
            end_token = min(start_token + self.chunk_tokens, total_tokens)
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = encoding.decode(chunk_tokens)

            # Calculate character position in original document
            # This is approximate but good enough for reference
            if seq == 0:
                pos = 0
            else:
                # Decode tokens up to this point to get character position
                pos = len(encoding.decode(tokens[:start_token]))

            chunks.append(
                Chunk(
                    seq=seq,
                    pos=pos,
                    content=chunk_text,
                    tokens=len(chunk_tokens),
                    bytes=len(chunk_text.encode("utf-8")),
                )
            )

            seq += 1
            start_token += step

            # Prevent infinite loop if step is 0 or negative
            if step <= 0:
                logger.error("Invalid chunking step size")
                break

        logger.debug(f"Chunked {total_tokens} tokens into {len(chunks)} chunks")
        return chunks

    def _chunk_by_chars(self, content: str) -> list[Chunk]:
        """Chunk content by character count with overlap (fallback)."""
        total_chars = len(content)

        # If content fits in a single chunk, return as-is
        if total_chars <= self.chunk_chars:
            return [
                Chunk(
                    seq=0,
                    pos=0,
                    content=content,
                    tokens=None,  # Unknown without tokenizer
                    bytes=len(content.encode("utf-8")),
                )
            ]

        chunks = []
        step = self.chunk_chars - self.overlap_chars
        seq = 0
        start_char = 0

        while start_char < total_chars:
            end_char = min(start_char + self.chunk_chars, total_chars)

            # Try to break at word boundary
            if end_char < total_chars:
                # Look for a space within the last 100 chars
                for i in range(end_char, max(start_char + step, end_char - 100), -1):
                    if content[i] in " \n\t":
                        end_char = i + 1
                        break

            chunk_text = content[start_char:end_char]

            chunks.append(
                Chunk(
                    seq=seq,
                    pos=start_char,
                    content=chunk_text,
                    tokens=None,
                    bytes=len(chunk_text.encode("utf-8")),
                )
            )

            seq += 1
            start_char += step

            if step <= 0:
                logger.error("Invalid chunking step size")
                break

        logger.debug(f"Chunked {total_chars} chars into {len(chunks)} chunks")
        return chunks

    def estimate_chunks(self, content: str) -> int:
        """Estimate how many chunks a document will produce.

        Useful for progress reporting without actually chunking.
        """
        encoding = self._get_encoding()
        if encoding:
            total_tokens = len(encoding.encode(content))
            if total_tokens <= self.chunk_tokens:
                return 1
            step = self.chunk_tokens - self.overlap_tokens
            return max(1, (total_tokens + step - 1) // step)
        else:
            total_chars = len(content)
            if total_chars <= self.chunk_chars:
                return 1
            step = self.chunk_chars - self.overlap_chars
            return max(1, (total_chars + step - 1) // step)


# Module-level default chunker instance
_default_chunker: Chunker | None = None


def get_chunker() -> Chunker:
    """Get the default chunker instance."""
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = Chunker()
    return _default_chunker


def chunk_document(content: str) -> list[Chunk]:
    """Convenience function to chunk a document using default settings."""
    return get_chunker().chunk(content)
