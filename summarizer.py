"""Gemini-backed LangChain chains for summarizing PDF text."""

from __future__ import annotations

import logging
import os
import re
import time

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, get_style

logger = logging.getLogger(__name__)
load_dotenv()

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_TEXT = (
    "429",
    "rate limit",
    "resource exhausted",
    "temporarily unavailable",
    "deadline exceeded",
    "timeout",
    "timed out",
    "500 internal",
    "502 bad gateway",
    "503 service",
    "504 gateway",
)


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient Gemini/API failures worth retrying."""
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUS:
        return True
    message = str(exc).lower()
    return any(fragment in message for fragment in _RETRYABLE_TEXT)


def _retry_invoke(chain, payload: dict, *, attempts: int = 3, base_delay: float = 1.0) -> str:
    """Invoke a chain with exponential backoff for transient errors."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return chain.invoke(payload)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == attempts - 1 or not _is_retryable(exc):
                raise
            delay = base_delay * (2**attempt)
            logger.warning("Transient LLM error; retrying in %.1fs: %s", delay, exc)
            time.sleep(delay)
    raise last_exc  # pragma: no cover - defensive


class PDFSummarizer:
    """Build and run the map/reduce summarization chains."""

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.3,
        max_concurrency: int = 4,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env and restart.")

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.max_concurrency = max_concurrency
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=temperature,
            google_api_key=api_key,
        )

        # Gemini can return content as a list of blocks; .text joins them.
        parser = RunnableLambda(lambda message: str(message.text))

        self.chunk_chain = CHUNK_SUMMARY | self.llm | parser
        self.combine_chain = COMBINE | self.llm | parser
        self.analysis_chain = ANALYSIS | self.llm | parser
        self.quotes_chain = QUOTES | self.llm | parser

    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        """Summarize a single text block."""
        style = get_style(summary_type)
        return _retry_invoke(self.chunk_chain, {"style": style, "text": text})

    def summarize_chunks(self, chunks: list[str], summary_type: str = "detailed") -> dict:
        """Summarize chunks in parallel, then combine them into one final summary."""
        style = get_style(summary_type)
        inputs = [{"style": style, "text": chunk} for chunk in chunks]

        raw_results = self.chunk_chain.batch(
            inputs,
            config={"max_concurrency": self.max_concurrency},
            return_exceptions=True,
        )

        results: list[str | Exception] = []
        for payload, result in zip(inputs, raw_results):
            if isinstance(result, Exception) and _is_retryable(result):
                try:
                    result = _retry_invoke(self.chunk_chain, payload)
                except Exception as exc:  # noqa: BLE001
                    result = exc
            results.append(result)

        summaries: list[dict] = []
        for index, (chunk, result) in enumerate(zip(chunks, results), start=1):
            ok = not isinstance(result, Exception)
            summary = result if ok else f"Error processing chunk: {result}"
            summaries.append(
                {
                    "chunk_number": index,
                    "summary": summary,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(summary) if ok else 0,
                }
            )

        return {
            "individual_summaries": summaries,
            "combined_summary": self.combine_summaries(summaries, summary_type),
            "total_chunks": len(chunks),
            "summary_type": summary_type,
        }

    def combine_summaries(self, chunk_summaries: list[dict], summary_type: str) -> str:
        """Merge successful chunk summaries into the final answer."""
        valid = [item["summary"] for item in chunk_summaries if item.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(
            f"Section {index}: {summary}" for index, summary in enumerate(valid, start=1)
        )
        style = get_style(summary_type)
        try:
            return _retry_invoke(
                self.combine_chain, {"style": style, "summaries": sections}
            )
        except Exception:  # noqa: BLE001
            return f"Combined Summary:\n\n{sections}"

    def analyze_document_structure(self, text: str) -> dict:
        """Return a short structure analysis, using only the first 3000 characters."""
        try:
            analysis = _retry_invoke(self.analysis_chain, {"text": text[:3000]})
            return {"analysis": analysis, "status": "success"}
        except Exception as exc:  # noqa: BLE001
            return {"analysis": f"Error analyzing document: {exc}", "status": "error"}

    def extract_key_quotes(self, text: str) -> list[str]:
        """Extract up to 10 verbatim quotes from the first 5000 characters."""
        response = _retry_invoke(self.quotes_chain, {"text": text[:5000]})
        lines = (
            re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip(' "“”')
            for line in response.splitlines()
        )
        return [line for line in lines if line][:10]
