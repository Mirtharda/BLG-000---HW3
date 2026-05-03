# Production Deployment Recommendation

**Project:** Local Wikipedia RAG Assistant  
**Course:** BLG483E HW3

---

## 1. Current State (Prototype)

The prototype is intentionally minimal: everything runs on a single laptop,
models are served by Ollama, and the vector store is a local ChromaDB file.
This is fine for a course project but has hard limits for real users.

---

## 2. Recommended Production Stack

### 2.1 Language Model

**Replace Ollama with a hosted or self-hosted inference server.**

| Option | When to use |
|--------|-------------|
| **vLLM** (self-hosted) | You control the hardware; need high throughput |
| **llama.cpp server** | Single-server, cost-sensitive deployment |
| **Anthropic / OpenAI API** | Fastest time-to-market; acceptable to use external APIs |
| **Amazon Bedrock / Azure OpenAI** | Enterprise compliance requirements |

For a real product, `llama3.2 3B` should be upgraded to **Llama-3 8B or 70B**
(or equivalent) for materially better answer quality.  With vLLM on a single
A100 GPU, a 70B model answers in 2–5 seconds at scale.

### 2.2 Embedding Model

`all-MiniLM-L6-v2` is good for a prototype but can be improved:

- **`nomic-embed-text`** (via Ollama or HuggingFace) — 8k context, outperforms MiniLM on retrieval benchmarks.
- **`text-embedding-3-small`** (OpenAI) — if external APIs are acceptable.
- For multilingual support: **`multilingual-e5-large`**.

### 2.3 Vector Database

| Option | When to use |
|--------|-------------|
| **Qdrant** | Self-hosted, production-grade, Rust-based, fast |
| **Weaviate** | Built-in hybrid search (BM25 + vectors) |
| **Pinecone** | Fully managed, minimal ops overhead |
| **pgvector** (PostgreSQL extension) | Already use Postgres; want SQL + vectors in one place |

For this specific use case (tens of millions of Wikipedia chunks), **Qdrant**
is the recommended choice: it supports payload filtering (replacing our
metadata approach), is horizontally scalable, and has a well-maintained
Python client.

### 2.4 Document Storage

Replace SQLite with **PostgreSQL** for:
- Concurrent writes during continuous ingestion.
- Full-text search fallback alongside vector search.
- Proper connection pooling.

### 2.5 API Layer

Wrap the RAG pipeline behind a **FastAPI** service:
```
POST /query    { question, n_results?, model? }  →  { answer, sources }
GET  /health   →  { ollama, vector_store, db }
```

This decouples the UI from the model server and enables horizontal scaling.

### 2.6 Frontend

Replace the Streamlit prototype with **Next.js + Tailwind CSS** for:
- Streaming token display (`EventSource` / SSE).
- Proper session management and authentication.
- Mobile-responsive layout.

---

## 3. Scalability Considerations

### Data Volume
- The current 40-article prototype has ~3,000 chunks.
- Full English Wikipedia is ~7 million articles → ~500 million chunks.
- At that scale: use **Qdrant distributed mode** or **Pinecone** with
  namespace-based partitioning by entity type.

### Concurrent Users
- vLLM's continuous batching handles multiple concurrent generation requests
  efficiently on GPU.
- The embedding model can be replicated behind a load balancer if needed
  (sentence-transformers is stateless).

### Freshness
- Schedule periodic re-ingestion from the Wikipedia API (daily or weekly).
- Use a job queue (Celery + Redis) to process new/updated articles.
- Incremental upsert to the vector store avoids full rebuilds.

---

## 4. Retrieval Quality Improvements

The current single-stage cosine-similarity retrieval works but can be improved:

1. **Hybrid search** — combine BM25 (keyword) and dense vector scores using
   Reciprocal Rank Fusion (RRF).  This helps for exact-match queries
   ("What year was Nikola Tesla born?") where dense retrieval can miss exact
   number matches.

2. **Re-ranking** — add a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
   as a second-stage ranker to re-score the top-20 retrieved chunks before
   passing the top-5 to the LLM.

3. **Query expansion** — use the LLM to generate 2–3 rephrasings of the user
   query, embed all of them, and take the union of retrieved chunks.

4. **Metadata enrichment** — store richer metadata (birth year, nationality,
   field of work for people; country, type, UNESCO status for places) to
   enable structured filtering alongside semantic search.

---

## 5. Production Monitoring

| What to monitor | Tool |
|----------------|------|
| LLM latency p50/p95 | Prometheus + Grafana |
| Retrieval quality (MRR, NDCG) | Offline eval suite with labelled queries |
| Answer quality (hallucination rate) | LLM-as-judge evaluation pipeline |
| Vector store health | Qdrant built-in metrics endpoint |
| API error rates | Sentry or Datadog |

---

## 6. Security

- Never expose the Ollama HTTP port directly to the internet.
- Add authentication (JWT or API key) to the FastAPI gateway.
- Rate-limit the `/query` endpoint to prevent abuse.
- If using Wikipedia API in production, cache responses and respect the
  Wikimedia User-Agent policy.

---

## 7. Cost Estimate (Self-Hosted)

| Component | Hardware | Monthly cost (AWS) |
|-----------|----------|-------------------|
| LLM server (llama3.2 70B) | g5.4xlarge (1× A10G) | ~$400 |
| Embedding server | c5.2xlarge (CPU) | ~$100 |
| Qdrant vector DB | r6i.xlarge (32 GB RAM) | ~$150 |
| PostgreSQL (RDS) | db.t3.medium | ~$50 |
| API + frontend | t3.medium | ~$30 |
| **Total** | | **~$730 / month** |

Using managed services (Pinecone + OpenAI) reduces ops burden but increases
per-query cost to ~$0.002–0.01 per query at moderate volume.

---

## 8. Summary Recommendation

For a production-grade version of this system:

1. Replace Ollama with **vLLM** serving **Llama-3 8B** on a single GPU node.
2. Swap ChromaDB for **Qdrant** with hybrid (dense + BM25) search.
3. Upgrade embeddings to **nomic-embed-text**.
4. Add a **FastAPI** service layer and a **Next.js** frontend.
5. Schedule **weekly re-ingestion** from Wikipedia with incremental upserts.
6. Add a **cross-encoder re-ranker** for 10–15% retrieval quality gain.

These changes would transform the course prototype into a system capable of
handling hundreds of concurrent users with high answer quality and low latency.
