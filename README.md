# HW3 — Local Wikipedia RAG Assistant

A fully local, ChatGPT-style question-answering system about famous people
and places. The system fetches Wikipedia articles, chunks and indexes them
with a local embedding model, and answers questions using a local LLM served
by Ollama — **no external APIs, no cloud calls at inference time**.

---

## Architecture Overview

```
Wikipedia API
     │
     ▼
ingest.py ──────► data/wikipedia.db  (SQLite, raw text)
                          │
                          ▼
               build_index.py
                ┌──────────────────────────────────────────┐
                │  chunker.py   — 800-char overlapping chunks │
                │  embedder.py  — all-MiniLM-L6-v2 (local)   │
                │  vector_store — ChromaDB (one collection)   │
                └──────────────────────────────────────────┘
                          │
                          ▼
                   data/chroma_db/
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
           app.py                   cli.py
       (Streamlit UI)            (terminal UI)
              │                        │
        retriever.py  ◄────────────────┘
        generator.py  → Ollama (llama3.2 / phi3 / mistral)
```

**Design choice — one vector store with metadata (Option B)**
A single ChromaDB collection stores all chunks with a `type` field
(`"person"` or `"place"`). This allows mixed/comparison queries to work
naturally by searching the full collection, while person-only or place-only
queries get a metadata filter for higher precision. Choosing Option B over
two separate collections avoids merging result sets and handles
comparison questions ("Compare Einstein and the Eiffel Tower") naturally.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.10 | runtime |
| [Ollama](https://ollama.com) | latest | local LLM server |

---

## 1 — Install Dependencies

```bash
cd hw3
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download `all-MiniLM-L6-v2`
> (~80 MB) on first use. After that, everything runs offline.

---

## 2 — Install & Start Ollama

**Windows:**
1. Download the installer from [https://ollama.com/download](https://ollama.com/download)
2. Run the `.exe` installer — Ollama starts automatically as a background service

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Pull the default model** (≈ 2 GB download, one time):
```bash
ollama pull llama3.2
```

Optional alternatives:
```bash
ollama pull phi3
ollama pull mistral
```

**Start the server** (skip on Windows — it runs automatically after install):
```bash
ollama serve
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
```

---

## 3 — Ingest Wikipedia Data

```bash
python ingest.py
```

Fetches full Wikipedia articles for 20 famous people and 20 famous places
and stores them in `data/wikipedia.db`.

Options:
```bash
python ingest.py --reset   # wipe DB and re-fetch everything
python ingest.py --list    # print what is currently stored
```

---

## 4 — Build the Vector Index

```bash
python build_index.py
```

Chunks every document, generates local embeddings, and stores them in
`data/chroma_db/`. Takes **2–5 minutes** on CPU.

```bash
python build_index.py --reset   # rebuild from scratch
```

---

## 5 — Start the Application

### Option A — Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens automatically at **http://localhost:8501**.

### Option B — CLI

```bash
python cli.py                        # default model (llama3.2)
python cli.py --model phi3           # use a different model
python cli.py --top-k 8              # retrieve more chunks
python cli.py --show-sources         # print retrieved passages
```

---

## Example Queries

### People
```
Who was Albert Einstein and what is he known for?
What did Marie Curie discover?
Why is Nikola Tesla famous?
Compare Lionel Messi and Cristiano Ronaldo.
What is Frida Kahlo known for?
```

### Places
```
Where is the Eiffel Tower located?
Why is the Great Wall of China important?
What was the Colosseum used for?
Where is Mount Everest?
What is Machu Picchu?
```

### Mixed / Comparison
```
Which famous place is located in Turkey?
Which person is associated with electricity?
Compare Albert Einstein and Nikola Tesla.
Compare the Eiffel Tower and the Statue of Liberty.
```

### Failure Cases (graceful "I don't know")
```
Who is the president of Mars?
Tell me about John Doe.
```

---

## Project Structure

```
hw3/
├── ingest.py           Wikipedia → SQLite ingestion
├── build_index.py      SQLite → ChromaDB indexing
├── app.py              Streamlit chat UI
├── cli.py              Terminal chat interface
├── requirements.txt
├── README.md
├── product_prd.md
├── recommendation.md
└── rag/
    ├── __init__.py
    ├── chunker.py       800-char overlapping chunker with sentence snapping
    ├── embedder.py      sentence-transformers wrapper (all-MiniLM-L6-v2)
    ├── vector_store.py  ChromaDB wrapper
    ├── retriever.py     keyword query classifier + cosine similarity search
    └── generator.py     Ollama LLM wrapper (blocking + streaming)
```

---

## Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM | Ollama llama3.2 3B | Runs on laptop CPU; good instruction following |
| Embeddings | all-MiniLM-L6-v2 | Fast, 80 MB, no API, strong semantic similarity |
| Vector DB | ChromaDB (persistent) | Zero-config local DB; Python-native API |
| Metadata strategy | Option B (one collection) | Enables cross-type queries without result merging |
| Chunking | 800 chars / 150 overlap | Richer context per chunk; overlap prevents boundary gaps |
| Query classification | Keyword matching | Zero latency; avoids an extra LLM call for routing |
| Raw storage | SQLite | Simple, standard; lets us re-chunk without re-fetching |

---

## Links

- **GitHub Repository:** _[(https://github.com/Mirtharda/BLG-000---HW3.git)]_
- **Demo Video:** _[Add your Loom / YouTube link here]_
