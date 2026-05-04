"""CLI chat interface — Local Wikipedia RAG Assistant.

Usage
-----
    python cli.py
    python cli.py --model phi3
    python cli.py --top-k 8 --show-sources

In-chat commands
----------------
    /sources   toggle display of retrieved chunks
    /clear     clear the terminal screen
    /quit      exit
"""
from __future__ import annotations

import argparse
import os

from rag.generator    import check_ollama, generate, DEFAULT_MODEL
from rag.retriever    import retrieve, classify_query
from rag.vector_store import collection_stats, get_collection

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        Local Wikipedia RAG Assistant  —  BLG000 HW3        ║
╚══════════════════════════════════════════════════════════════╝
Type your question and press Enter.
Commands:  /sources  /clear  /quit
"""

_TYPE_LABEL = {
    "person": "👤 person",
    "place":  "📍 place",
    "both":   "🔀 mixed",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia RAG CLI")
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help="Ollama model name (default: llama3.2)")
    parser.add_argument("--top-k",        type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--show-sources", action="store_true",
                        help="Print retrieved chunks before the answer")
    args = parser.parse_args()

    print(BANNER)

    # ── Startup checks ────────────────────────────────────────────────────
    ollama_ok, models = check_ollama()
    if ollama_ok:
        model_list = ", ".join(models[:6]) if models else "(none pulled yet)"
        print(f"  ✓  Ollama running  |  available: {model_list}")
    else:
        print("  ⚠  Ollama not detected — start it with:  ollama serve")

    try:
        col   = get_collection()
        stats = collection_stats(col)
        print(f"  ✓  Vector store ready  |  {stats['total_chunks']:,} chunks indexed")
    except Exception:
        print("  ⚠  Vector store not found — run:  python build_index.py")

    print(f"\n  Model: {args.model}   Top-k: {args.top_k}\n")

    show_sources: bool = args.show_sources

    # ── Chat loop ─────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        if user_input.lower() == "/sources":
            show_sources = not show_sources
            print(f"  [sources display: {'ON' if show_sources else 'OFF'}]\n")
            continue
        if user_input.lower() == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        # Classify + retrieve
        query_type    = classify_query(user_input)
        chunks, _     = retrieve(user_input, n_results=args.top_k)
        type_str      = _TYPE_LABEL.get(query_type, query_type)

        print(f"\n  [{type_str} query  ·  {len(chunks)} chunks retrieved]\n")

        # Optionally print source chunks
        if show_sources and chunks:
            print("  ── Retrieved sources ──────────────────────────────────")
            for i, c in enumerate(chunks, 1):
                meta  = c.get("metadata", {})
                sim   = round(1.0 - c.get("distance", 1.0), 3)
                print(f"  [{i}] {meta.get('entity','?')} ({meta.get('type','?')})  "
                      f"sim={sim}")
                preview = c["text"].replace("\n", " ")
                print(f"       {preview[:130]}…")
            print("  ──────────────────────────────────────────────────────\n")

        # Generate
        print("Assistant: ", end="", flush=True)
        answer = generate(user_input, chunks, model=args.model)
        print(answer)
        print()


if __name__ == "__main__":
    main()
