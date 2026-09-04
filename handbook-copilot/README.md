# 📘 Handbook Copilot

Local-first RAG over your PDFs. Drop documents into a folder, ingest them once,
then ask questions in a chat UI. Answers are grounded in the documents with
inline `[source, p.N, §section]` citations, and the retrieved excerpts are
shown alongside every reply.

Runs entirely on your machine by default (embeddings + vector DB + local LLM
via Ollama). Optionally switch the answer model to Claude or GPT with an API
key — only the question and retrieved excerpts leave your machine, never the
raw PDFs.

## Stack

- **Parsing:** PyMuPDF (page-accurate text)
- **Chunking:** heading-aware, ~600 tokens with 100-token overlap
- **Embeddings:** `bge-small-en-v1.5` (local, 384-dim)
- **Vector DB:** ChromaDB (embedded, persisted to `data/chroma/`)
- **Reranker:** `bge-reranker-base` cross-encoder
- **LLM (pluggable):** Ollama (default) · Anthropic · OpenAI
- **UI:** Streamlit

## Setup

### 1. Python env

```bash
cd handbook-copilot
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

First run downloads two Hugging Face models (~150 MB total) into the local cache.

### 2. Install Ollama (for the default local LLM)

Download the Windows installer from <https://ollama.com/download> and run it.
Then, in a new terminal:

```bash
ollama pull llama3.1:8b
ollama serve
```

`ollama serve` runs in the background at `http://localhost:11434`. You can
close the window once it's running; Windows keeps the service alive.

If you'd rather skip local models and only use API providers, that's fine —
just paste an API key in the app's sidebar when it opens.

### 3. Configure (optional)

```bash
cp .env.example .env
# edit .env if you want to hard-code API keys instead of pasting them in the UI
```

## Use it

```bash
streamlit run app.py
```

Streamlit opens `http://localhost:8501` in your browser.

**In the sidebar:**
1. Drag & drop one or more PDFs into the uploader.
2. Click **Ingest uploaded files** — the app parses, chunks, embeds, and
   indexes them with a live progress bar per file. Uploaded PDFs are also
   saved to `data/pdfs/` so nothing is lost between runs.
3. Pick a provider (Ollama is the default — no key needed once Ollama is
   running) and ask questions in the chat.

The **Sources** expander shows every indexed file with a ✕ button to remove
one. Re-uploading a PDF with the same filename is a no-op unless you tick
**Re-ingest if already present**.

### CLI (optional, still works)

For bulk ingestion or scripting, the CLI takes a file or directory:

```bash
python ingest.py data/pdfs           # skip files already indexed
python ingest.py data/pdfs --force   # re-ingest everything
python ingest.py data/pdfs --reset   # wipe the DB first
```

## Project layout

```
handbook-copilot/
├── app.py              # Streamlit chat UI
├── ingest.py           # CLI: parse → chunk → embed → store
├── core/
│   ├── parser.py       # PDF → pages
│   ├── chunker.py      # heading-aware splitting
│   ├── embedder.py     # bge-small-en-v1.5
│   ├── store.py        # ChromaDB wrapper
│   ├── retriever.py    # search + rerank
│   ├── prompts.py      # citation-enforcing prompt
│   └── llm/            # Ollama / Anthropic / OpenAI providers
└── data/
    ├── pdfs/           # your source documents (gitignored)
    └── chroma/         # persistent vector DB (gitignored)
```

## Design notes

- **Citations are metadata, not prose.** Every chunk carries `(source, page,
  section_path)`. The prompt tells the model to emit those exact tokens; the
  UI additionally renders the raw excerpts so you can verify any answer.
- **Heading-aware chunking** splits on numbered sections (`1.`, `2.3.1`,
  `Chapter 4`) before falling back to recursive character splitting. Keeps
  semantic units intact so citations land on real sections, not mid-paragraph.
- **Reranking is the biggest quality lever** after chunking. We retrieve
  top-20 by vector similarity and rerank down to top-5 with a cross-encoder.
- **Provider abstraction is deliberately tiny** — one `generate(messages)`
  method that streams strings. Adding a new backend is one file.

## Troubleshooting

- **`Ollama isn't reachable`** — run `ollama serve` in another terminal, or
  set `OLLAMA_HOST` in `.env`.
- **Slow first ingest** — the embedding model downloads on first use. After
  that, ingesting is CPU-bound (~500 chunks/min on a laptop CPU).
- **Wrong page numbers** — some PDFs use logical page numbers different from
  physical ones; PyMuPDF reports the physical (0-indexed → we display 1-indexed)
  page. This matches what most viewers show in the page counter.
