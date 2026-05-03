"""ChromaDB vector store wrapper.

Design choice: Option B — one collection with metadata
------------------------------------------------------
A single collection stores all chunks (people + places) with a
``type`` metadata field ("person" or "place").

Rationale over Option A (two collections):
* Mixed / comparison queries ("Compare Einstein and Eiffel Tower") work
  naturally without merging two result sets.
* The retriever can filter by type when the query is clearly about one
  category, and fall back to full-collection search for ambiguous queries.
* One client, one collection → simpler code and a single index file.
* ChromaDB metadata filtering is efficient; the overhead of one combined
  collection vs. two separate ones is negligible at this data size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

CHROMA_PATH     = Path("data/chroma_db")
COLLECTION_NAME = "wikipedia_rag"


def _client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection(reset: bool = False) -> chromadb.Collection:
    """Return (or create) the shared collection.  Pass reset=True to rebuild."""
    client = _client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )


def add_chunks(
    collection: chromadb.Collection,
    ids:        list[str],
    embeddings: list[list[float]],
    documents:  list[str],
    metadatas:  list[dict[str, Any]],
    batch_size: int = 100,
) -> None:
    """Upsert chunks in batches to avoid memory spikes on large ingestions."""
    for i in range(0, len(ids), batch_size):
        sl = slice(i, i + batch_size)
        collection.upsert(
            ids=ids[sl],
            embeddings=embeddings[sl],
            documents=documents[sl],
            metadatas=metadatas[sl],
        )


def query_store(
    collection:      chromadb.Collection,
    query_embedding: list[float],
    n_results:       int = 5,
    where:           dict | None = None,
) -> list[dict]:
    """Semantic search.  Returns list of dicts with keys: text, metadata, distance."""
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        )
    ]


def collection_stats(collection: chromadb.Collection) -> dict:
    return {
        "collection":   COLLECTION_NAME,
        "total_chunks": collection.count(),
    }
