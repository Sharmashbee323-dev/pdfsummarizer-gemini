"""PDF parsing, cleanup and token-aware chunking for the summarizer."""

from __future__ import annotations

import logging
import re
from typing import BinaryIO

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Turn an uploaded PDF into clean, token-limited text chunks."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load tokenizer {encoding_name!r}: {exc}. "
                "Check your connection and reinstall requirements.txt."
            ) from exc

    def extract_text_from_pdf(self, pdf_file: BinaryIO) -> str:
        """Return all page text. A page that cannot be extracted is skipped."""
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Error reading PDF: {exc}") from exc

        pages: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping page %d: %s", page_number, exc)

        return "\n\n".join(pages)

    def clean_text(self, text: str) -> str:
        """Remove control characters and extra whitespace, keeping punctuation and paragraphs."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Approximate Gemini token count using the cl100k_base tokenizer."""
        return len(self.encoding.encode(text, disallowed_special=()))

    def chunk_text(self, text: str, max_tokens: int = 8000) -> list[str]:
        """Split text into chunks of at most ``max_tokens``, preferring sentence boundaries."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if not text.strip():
            raise ValueError("No extractable text — the PDF may be scanned images")

        pieces: list[tuple[str, int]] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            tokens = self.encoding.encode(sentence, disallowed_special=())
            if len(tokens) > max_tokens:
                pieces.extend(
                    (
                        self.encoding.decode(tokens[start : start + max_tokens]),
                        min(max_tokens, len(tokens) - start),
                    )
                    for start in range(0, len(tokens), max_tokens)
                )
            else:
                pieces.append((sentence, len(tokens)))

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for piece, token_count in pieces:
            if current and current_tokens + token_count + 1 > max_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += token_count + 1  # +1 for the joining space

        if current:
            chunks.append(" ".join(current))
        return chunks

    def get_pdf_metadata(self, pdf_file: BinaryIO) -> dict:
        """Return page count, title, author and subject, or ``{'error': ...}``."""
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            metadata = reader.metadata or {}

            def field(key: str) -> str:
                return metadata.get(key) or "Unknown"

            return {
                "num_pages": len(reader.pages),
                "title": field("/Title"),
                "author": field("/Author"),
                "subject": field("/Subject"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
