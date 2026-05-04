# Product Requirements Document
## Local Wikipedia RAG Assistant

**Course:** BLG000 — Artificial Intelligence  
**Project:** HW3  
**Version:** 1.0

---

## 1. Problem Statement

Users want factual answers about well-known people and places without
relying on cloud AI services.  Existing solutions require internet access,
API subscriptions, and send private queries to remote servers.  This
project delivers a fully local alternative: a conversational assistant
grounded in Wikipedia content, running entirely on a personal laptop.

---

## 2. Goals

1. Answer natural-language questions about 20+ famous people and 20+ famous places.
2. Ground every answer in retrieved Wikipedia content — no hallucinated facts.
3. Run entirely on localhost: no external LLM API, no cloud vector DB.
4. Provide a usable chat interface (Streamlit UI + CLI).
5. Handle gracefully the case where the answer is not in the knowledge base.

---

## 3. Non-Goals

- Real-time Wikipedia updates (knowledge is static after ingestion).
- Answering questions outside the ingested entity set.
- Multi-language support.
- Production scalability (this is a research/course prototype).

---

## 4. Users

**Primary:** Computer Engineering students and researchers evaluating local RAG systems.  
**Secondary:** Instructors running and grading the system.

---

## 5. Functional Requirements

### 5.1 Ingestion
- FR-1: Fetch full Wikipedia articles for ≥ 20 people and ≥ 20 places.
- FR-2: Store raw text and metadata in a local SQLite database.
- FR-3: Support incremental re-ingestion (skip already-fetched entries).
- FR-4: Support full reset and re-ingestion via a CLI flag.

### 5.2 Chunking
- FR-5: Split documents into overlapping fixed-size character chunks (500 chars, 100 overlap).
- FR-6: Snap chunk boundaries to sentence endings when possible.
- FR-7: Discard degenerate chunks shorter than 80 characters.

### 5.3 Embedding & Indexing
- FR-8: Generate embeddings using a local model (all-MiniLM-L6-v2).
- FR-9: Persist vectors in ChromaDB with entity name and type metadata.
- FR-10: Support idempotent upsert so re-indexing doesn't duplicate data.

### 5.4 Retrieval
- FR-11: Classify each query as person, place, or mixed via keyword matching.
- FR-12: Apply a metadata filter when the query is clearly one type.
- FR-13: Fall back to full-collection search when fewer than 2 filtered results are found.
- FR-14: Return the top-k most similar chunks (configurable, default 5).

### 5.5 Generation
- FR-15: Build a grounded prompt from retrieved chunks + user query.
- FR-16: Instruct the model to answer only from context.
- FR-17: Return "I don't know based on the available information." when context is absent.
- FR-18: Support Ollama models: llama3.2, phi3, mistral.

### 5.6 Chat Interface
- FR-19: Streamlit UI with chat history, model selector, and source viewer.
- FR-20: CLI with the same capabilities and `/sources`, `/clear`, `/quit` commands.
- FR-21: System status display (Ollama health, vector store chunk count).
- FR-22: Conversation reset button / command.

---

## 6. Non-Functional Requirements

- NFR-1: First answer (after index is built) in under 30 seconds on a mid-range laptop CPU.
- NFR-2: Index build time under 10 minutes for 40 documents on CPU.
- NFR-3: All data stored locally; no network calls at inference time.
- NFR-4: Single `pip install -r requirements.txt` installs all Python dependencies.
- NFR-5: Instructor can run the full system following only README instructions.

---

## 7. Data Model

### SQLite — `documents` table

| Column     | Type    | Description                          |
|------------|---------|--------------------------------------|
| id         | INTEGER | Primary key                          |
| title      | TEXT    | Wikipedia page title (unique)        |
| type       | TEXT    | `"person"` or `"place"`             |
| content    | TEXT    | Full extracted plain text            |
| url        | TEXT    | Canonical Wikipedia URL              |
| char_count | INTEGER | Length of content                    |
| fetched_at | TEXT    | ISO timestamp of ingestion           |

### ChromaDB — `wikipedia_rag` collection

| Field      | Type          | Description                         |
|------------|---------------|-------------------------------------|
| id         | string        | `{entity}__{type}__{chunk_index}`  |
| document   | string        | Chunk text                          |
| embedding  | float[384]    | all-MiniLM-L6-v2 vector             |
| metadata   | dict          | entity, type, chunk_index, source_url |

---

## 8. System Flow

```
User query
    │
    ▼
classify_query()      ← keyword matching against entity name sets
    │
    ▼
embed_one()           ← sentence-transformers local model
    │
    ▼
query_store()         ← ChromaDB cosine similarity search
    │              (with optional metadata filter)
    ▼
generate()            ← Ollama LLM, grounded prompt
    │
    ▼
Answer displayed in Streamlit / CLI
```

---

## 9. Acceptance Criteria

- All 10 required people and 10 required places return sensible answers to the example questions listed in the assignment.
- Out-of-domain queries ("Who is the president of Mars?") return "I don't know" rather than hallucinated answers.
- System starts and answers questions by following only the README.
- Source chunks are optionally displayable for every answer.
