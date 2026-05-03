"""Build the ChromaDB vector index from the SQLite database.

Reads all documents from data/wikipedia.db, chunks them with the
configured chunker, generates sentence-transformer embeddings in batches,
and upserts everything into the shared ChromaDB collection.

Usage
-----
    python build_index.py            # build / update index (idempotent upsert)
    python build_index.py --reset    # wipe collection first, then rebuild

Run this after ingest.py whenever the database changes.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from rag.chunker      import chunk_text, Chunk
from rag.embedder     import embed
from rag.vector_store import get_collection, add_chunks, collection_stats

DB_PATH    = Path("data/wikipedia.db")
EMBED_BATCH = 32    # texts per embedding batch (tune for available RAM)


def _safe_id(entity: str, entity_type: str, idx: int) -> str:
    """Create a stable, filesystem-safe chunk ID."""
    safe = entity.lower().replace(" ", "_").replace("/", "_").replace(".", "")
    return f"{safe}__{entity_type}__{idx}"


def build_index(reset: bool = False) -> None:
    if not DB_PATH.exists():
        print(f"[error] {DB_PATH} not found.  Run  python ingest.py  first.")
        return

    print(f"Loading documents from {DB_PATH}…")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT title, type, content, url FROM documents ORDER BY type, title"
    ).fetchall()
    conn.close()
    print(f"  Found {len(rows)} documents.\n")

    # ── Chunking ──────────────────────────────────────────────────────────────
    all_chunks: list[Chunk] = []
    for title, etype, content, url in rows:
        chunks = chunk_text(
            content,
            entity=title,
            entity_type=etype,
            source_url=url or "",
        )
        all_chunks.extend(chunks)
        print(f"  {title:<35} → {len(chunks):>3} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # ── Embedding ─────────────────────────────────────────────────────────────
    print(f"\nGenerating embeddings (batch size = {EMBED_BATCH})…")
    ids:        list[str]         = []
    texts:      list[str]         = []
    metadatas:  list[dict]        = []
    embeddings: list[list[float]] = []

    total_batches = (len(all_chunks) + EMBED_BATCH - 1) // EMBED_BATCH
    for b_idx in range(0, len(all_chunks), EMBED_BATCH):
        batch      = all_chunks[b_idx : b_idx + EMBED_BATCH]
        batch_num  = b_idx // EMBED_BATCH + 1
        print(f"  batch {batch_num}/{total_batches}…", end=" ", flush=True)

        batch_texts = [c.text for c in batch]
        batch_embs  = embed(batch_texts)
        print("✓")

        for chunk, emb in zip(batch, batch_embs):
            ids.append(_safe_id(chunk.entity, chunk.entity_type, chunk.chunk_index))
            texts.append(chunk.text)
            embeddings.append(emb)
            metadatas.append({
                "entity":      chunk.entity,
                "type":        chunk.entity_type,
                "chunk_index": chunk.chunk_index,
                "source_url":  chunk.source_url,
            })

    # ── Store ─────────────────────────────────────────────────────────────────
    print(f"\nUpserting {len(ids)} chunks into ChromaDB…")
    collection = get_collection(reset=reset)
    add_chunks(collection, ids, embeddings, texts, metadatas)

    stats = collection_stats(collection)
    print(f"\n=== Index built ===")
    print(f"  Collection : {stats['collection']}")
    print(f"  Total chunks stored: {stats['total_chunks']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the existing collection before rebuilding.",
    )
    args = parser.parse_args()
    build_index(reset=args.reset)


if __name__ == "__main__":
    main()
