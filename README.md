# 📄 PDF AI Summarizer

A Streamlit app that summarizes PDF documents with **Google Gemini** through **LangChain**.
Upload a PDF, choose a summary style, and get a final summary plus optional document analysis and key quotes.

## What it does

- Extracts and cleans text from a PDF, then splits it into token-limited chunks.
- Summarizes each chunk in parallel, then combines them into one final summary (map/reduce).
- Supports four styles: `brief`, `detailed`, `bullet_points`, `executive`.
- Optionally analyzes document structure and extracts key quotes.
- Exports the report as `.txt` and per-chunk statistics as `.csv`.

## Project structure

```text
app.py            Streamlit UI: settings, upload, results and downloads
pdf_processor.py  PDF metadata, text extraction, cleaning, token counting and chunking
prompts.py        Shared system prompt, styles and four ChatPromptTemplates
summarizer.py     Gemini/LangChain chains, retries, batching and combine logic
utils.py          Display helpers, exports and API-key check
.env.example      Environment variable template
requirements.txt  Python dependencies
smoke_test.py     Small end-to-end check that calls Gemini
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Get a free API key from <https://aistudio.google.com/apikey> and set it in `.env`:

```dotenv
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

`.env` is gitignored. Never commit your key.

## Run

```bash
streamlit run app.py
```

## Smoke test

With the same virtual environment active:

```bash
python smoke_test.py
```

This creates a tiny in-memory PDF, extracts its text, and runs one `brief` summary through the
full LangChain pipeline. It makes a real (small) Gemini API call.

## How it works

```text
PDF → extract text → clean → chunk by tokens
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
             summary 1       summary 2   ...  summary N     (map, 4 at a time)
                  └───────────────┼───────────────┘
                                  ▼
                        combine → final summary            (reduce)
```

## Adding a style

1. Add an entry to `STYLES` in `prompts.py`.
2. Nothing else is needed: `app.py` reads the style names from `prompts.py`.

Unknown style names raise a clear `ValueError` instead of silently falling back.

## Troubleshooting

- **No extractable text**: the PDF is probably a scanned image. Run OCR first.
- **429 / rate limit**: wait and retry; use a shorter document or a different model.
- **Model unavailable / 404**: set `GEMINI_MODEL` in `.env` to a model your key can access.
- **API key not found**: ensure `.env` is next to `app.py` and restart Streamlit after editing it.
