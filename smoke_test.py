"""Small end-to-end check: build a PDF, extract text, and call Gemini via LangChain."""

from __future__ import annotations

import io

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from pdf_processor import PDFProcessor
from summarizer import PDFSummarizer

TEXT = (
    "Renewable energy is growing quickly. Solar power is now one of the cheapest "
    "sources of new electricity. Wind power also plays an important role. Together, "
    "these technologies can reduce emissions and improve energy independence."
)


def make_tiny_pdf() -> io.BytesIO:
    """Create a one-page PDF with selectable text, without extra dependencies."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    content = DecodedStreamObject()
    content.set_data(f"BT /F1 11 Tf 72 720 Td ({TEXT}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )

    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


def main() -> None:
    processor = PDFProcessor()
    pdf = make_tiny_pdf()

    metadata = processor.get_pdf_metadata(pdf)
    raw_text = processor.extract_text_from_pdf(pdf)
    clean_text = processor.clean_text(raw_text)
    chunks = processor.chunk_text(clean_text, max_tokens=500)

    print("PAGES:", metadata.get("num_pages"))
    print("EXTRACTED_CHARS:", len(raw_text))
    print("CHUNKS:", len(chunks))

    if not chunks:
        raise RuntimeError("No chunks were produced")

    summarizer = PDFSummarizer()
    result = summarizer.summarize_text(chunks[0], "brief")
    print("SUMMARY:", result.strip()[:300])


if __name__ == "__main__":
    main()
