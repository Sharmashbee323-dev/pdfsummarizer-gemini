"""Streamlit UI for the Gemini/LangChain PDF summarizer."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from prompts import STYLE_NAMES
from summarizer import PDFSummarizer
from utils import (
    create_summary_dataframe,
    estimate_processing_time,
    export_summary_to_text,
    format_summary_display,
    validate_api_key,
)

load_dotenv()

MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
]

st.set_page_config(page_title="PDF AI Summarizer", page_icon="📄", layout="wide")


def main() -> None:
    st.title("📄 PDF Reader + AI Summarizer")
    st.markdown(
        "Upload a PDF document and get intelligent summaries powered by Google Gemini."
    )

    if not validate_api_key():
        st.stop()

    st.sidebar.header("⚙ Settings")

    default_model = os.getenv("GEMINI_MODEL", MODELS[0])
    models = MODELS if default_model in MODELS else [default_model] + MODELS
    model = st.sidebar.selectbox(
        "Model",
        models,
        index=models.index(default_model),
        help="Gemini model used for all LLM calls",
    )

    summary_type = st.sidebar.selectbox(
        "Summary Type",
        list(STYLE_NAMES),
        index=list(STYLE_NAMES).index("detailed"),
        help="Choose the type of summary you want",
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens per Chunk",
        2000,
        10000,
        8000,
        step=500,
        help="Maximum tokens per text chunk (affects processing speed)",
    )

    show_analysis = st.sidebar.checkbox(
        "Show Document Analysis",
        value=True,
        help="Analyze document structure and content",
    )
    show_quotes = st.sidebar.checkbox(
        "Extract Key Quotes",
        value=False,
        help="Extract important quotes from the document",
    )

    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()

    if st.session_state.get("model") != model:
        try:
            st.session_state.summarizer = PDFSummarizer(model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Could not initialize Gemini: {exc}")
            st.stop()
        st.session_state.model = model
        st.session_state.pop("results", None)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to summarize",
    )

    file_id = uploaded_file.file_id if uploaded_file else None
    if st.session_state.get("file_id") != file_id:
        st.session_state.pop("results", None)
        st.session_state.file_id = file_id

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        with st.spinner("Analyzing PDF metadata..."):
            metadata = st.session_state.pdf_processor.get_pdf_metadata(uploaded_file)

        if "error" in metadata:
            st.warning(f"⚠ Could not read PDF metadata: {metadata['error']}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📄 Pages: {metadata.get('num_pages', 'Unknown')}")
                st.info(f"📝 Title: {metadata.get('title', 'Unknown')}")
            with col2:
                st.info(f"👤 Author: {metadata.get('author', 'Unknown')}")
                st.info(f"📋 Subject: {metadata.get('subject', 'Unknown')}")

        if st.button("🚀 Process PDF", type="primary"):
            process_pdf(
                uploaded_file,
                summary_type,
                max_tokens,
                show_analysis,
                show_quotes,
            )

        if "results" in st.session_state:
            display_results(st.session_state.results)


def process_pdf(
    uploaded_file,
    summary_type: str,
    max_tokens: int,
    show_analysis: bool,
    show_quotes: bool,
) -> None:
    """Run the pipeline and store results in ``st.session_state['results']``."""
    st.session_state.pop("results", None)
    processor = st.session_state.pdf_processor
    summarizer = st.session_state.summarizer

    with st.spinner("📖 Extracting text from PDF..."):
        try:
            raw_text = processor.extract_text_from_pdf(uploaded_file)
            st.success(f"✅ Extracted {len(raw_text)} characters")
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Error extracting text: {exc}")
            return

    with st.spinner("🧹 Cleaning and processing text..."):
        clean_text = processor.clean_text(raw_text)
        token_count = processor.count_tokens(clean_text)
        st.success(f"✅ Cleaned text: {len(clean_text)} characters, ~{token_count} tokens")

    with st.spinner("✂ Splitting text into chunks..."):
        try:
            chunks = processor.chunk_text(clean_text, max_tokens)
        except ValueError as exc:
            st.error(f"❌ {exc}")
            return
        st.success(f"✅ Created {len(chunks)} chunks")
        st.info(f"⏱ Estimated processing time: {estimate_processing_time(len(chunks))}")

    analysis = None
    if show_analysis:
        with st.spinner("🔍 Analyzing document structure..."):
            analysis = summarizer.analyze_document_structure(clean_text)

    with st.spinner(f"🤖 Generating {summary_type} summaries..."):
        try:
            summary_data = summarizer.summarize_chunks(chunks, summary_type)
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Error generating summaries: {exc}")
            return

    quotes = None
    if show_quotes:
        with st.spinner("💬 Extracting key quotes..."):
            try:
                quotes = summarizer.extract_key_quotes(clean_text[:5000])
            except Exception:  # noqa: BLE001
                st.warning("⚠ Could not extract key quotes")

    st.session_state.results = {
        "file_name": uploaded_file.name,
        "summary_data": summary_data,
        "analysis": analysis,
        "quotes": quotes,
    }


def display_results(results: dict) -> None:
    """Render stored results: analysis, quotes, summary, downloads and stats."""
    summary_data = results["summary_data"]
    file_name = results["file_name"]

    analysis = results.get("analysis")
    if analysis:
        if analysis.get("status") == "success":
            st.subheader("📊 Document Analysis")
            st.write(analysis.get("analysis", ""))
        else:
            st.warning("⚠ Could not analyze document structure")

    if results.get("quotes"):
        st.subheader("💬 Key Quotes")
        for index, quote in enumerate(results["quotes"], start=1):
            st.write(f"{index}. *\"{quote}\"*")

    st.header("📋 Summary Results")
    format_summary_display(summary_data)

    st.subheader("📤 Export Options")
    summary_df = create_summary_dataframe(summary_data)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download as Text",
            data=export_summary_to_text(summary_data, file_name).encode("utf-8"),
            file_name=f"{file_name}_summary.txt",
            mime="text/plain",
        )
    with col2:
        st.download_button(
            label="📊 Download Statistics as CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{file_name}_stats.csv",
            mime="text/csv",
        )

    with st.expander("📈 Processing Statistics"):
        st.dataframe(summary_df)
        st.metric(
            "Average Compression Ratio",
            f"{summary_df['Compression Ratio'].mean():.2f}",
        )
        st.metric(
            "Success Rate",
            f"{(summary_df['Status'] == 'Success').mean() * 100:.1f}%",
        )


if __name__ == "__main__":
    main()
