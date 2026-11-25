"""Display helpers, exports and basic configuration checks for the Streamlit app."""

from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st


def format_summary_display(summary_data: dict) -> None:
    """Render statistics, the final summary and the individual chunk summaries."""
    st.subheader("📊 Summary Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", summary_data["total_chunks"])
    col2.metric(
        "Successful Summaries",
        sum(item["ok"] for item in summary_data["individual_summaries"]),
    )
    col3.metric("Summary Type", summary_data["summary_type"].replace("_", " ").title())

    st.subheader("📝 Final Summary")
    st.write(summary_data["combined_summary"])

    with st.expander("View Individual Chunk Summaries"):
        for item in summary_data["individual_summaries"]:
            suffix = "" if item["ok"] else " ⚠ failed"
            st.write(f"**Chunk {item['chunk_number']}**{suffix}")
            st.write(item["summary"])
            st.write(
                f"*Original length: {item['original_length']} chars, "
                f"Summary length: {item['summary_length']} chars*"
            )
            st.divider()


def export_summary_to_text(summary_data: dict, filename: str = "") -> str:
    """Build a plain-text report; ``filename`` appears in the header when supplied."""
    lines = ["PDF Summary Report", "==================", ""]
    if filename:
        lines.append(f"File: {filename}")
    lines += [
        f"Summary Type: {summary_data['summary_type'].replace('_', ' ').title()}",
        f"Total Chunks Processed: {summary_data['total_chunks']}",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "=============",
        "",
        summary_data["combined_summary"],
        "",
        "INDIVIDUAL CHUNK SUMMARIES",
        "==========================",
        "",
    ]
    for item in summary_data["individual_summaries"]:
        suffix = "" if item["ok"] else " (FAILED)"
        lines += [
            f"Chunk {item['chunk_number']}{suffix}:",
            item["summary"],
            "",
        ]
    return "\n".join(lines)


def create_summary_dataframe(summary_data: dict) -> pd.DataFrame:
    """Return one row per chunk with lengths, compression ratio and status."""
    return pd.DataFrame(
        [
            {
                "Chunk": item["chunk_number"],
                "Original Length": item["original_length"],
                "Summary Length": item["summary_length"],
                "Compression Ratio": (
                    round(item["summary_length"] / item["original_length"], 2)
                    if item["original_length"]
                    else 0
                ),
                "Status": "Success" if item["ok"] else "Error",
            }
            for item in summary_data["individual_summaries"]
        ]
    )


def estimate_processing_time(num_chunks: int) -> str:
    """Rough estimate: ~3s per chunk with 4 in parallel, plus 5s to combine."""
    seconds = int(num_chunks * 3 / 4 + 5)
    if seconds < 60:
        return f"~{seconds} seconds"
    return f"~{seconds // 60} min {seconds % 60}s"


def validate_api_key() -> bool:
    """Check that a plausible Gemini key is configured, without printing it."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        st.error(
            "❌ GEMINI_API_KEY not found! Copy `.env.example` to `.env`, add your key, and restart."
        )
        st.info("Get a free key at https://aistudio.google.com/apikey")
        return False
    if len(api_key) < 20:
        st.error(
            "❌ GEMINI_API_KEY seems incomplete. Get a new one at "
            "https://aistudio.google.com/apikey"
        )
        return False
    st.sidebar.caption("🔑 Gemini API key loaded")
    return True
