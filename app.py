"""Streamlit chat interface — Local Wikipedia RAG Assistant.

Usage
-----
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from rag.generator    import check_ollama, generate, generate_stream, DEFAULT_MODEL
from rag.retriever    import retrieve, classify_query
from rag.vector_store import collection_stats, get_collection

# ─── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wikipedia RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ─── Session state initialisation ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources?, query_type?}]

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox(
        "Ollama model",
        options=["llama3.2", "phi3", "mistral"],
        index=0,
        help="Model must be pulled via  ollama pull <name>",
    )

    n_results = st.slider(
        "Chunks to retrieve",
        min_value=3, max_value=10, value=5,
        help="3–5 recommended for CPU. More chunks = richer context but slower generation.",
    )

    show_sources = st.checkbox(
        "Show retrieved source chunks",
        value=False,
        help="Expand source passages used to generate each answer.",
    )

    st.divider()

    # ── System status ──────────────────────────────────────────────────────
    st.subheader("System Status")

    ollama_ok, available_models = check_ollama()
    if ollama_ok:
        st.success("✅ Ollama is running")
        if available_models:
            st.caption("Available: " + ", ".join(available_models[:6]))
    else:
        st.error("❌ Ollama not detected\n\nStart it with:\n```\nollama serve\n```")

    try:
        col   = get_collection()
        stats = collection_stats(col)
        st.success(f"✅ Vector store ready\n\n{stats['total_chunks']:,} chunks indexed")
    except Exception:
        st.error(
            "❌ Vector store empty\n\nRun:\n"
            "```\npython ingest.py\npython build_index.py\n```"
        )

    st.divider()

    if st.button("🗑️  Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("BLG000 HW3 · Local Wikipedia RAG")

# ─── Main chat area ────────────────────────────────────────────────────────────

st.title("📚 Local Wikipedia RAG Assistant")
st.caption(
    "Ask questions about famous people and places. "
    "All inference runs locally — no external APIs."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages if enabled
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            type_label = {
                "person": "👤 person query",
                "place":  "📍 place query",
                "both":   "🔀 mixed / open query",
            }
            qt = msg.get("query_type", "both")
            st.caption(
                f"{type_label.get(qt, qt)} · "
                f"{len(msg['sources'])} chunks retrieved"
            )
            with st.expander("📄 Retrieved source chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    meta       = src.get("metadata", {})
                    entity     = meta.get("entity", "—")
                    etype      = meta.get("type", "—")
                    similarity = round(1.0 - src.get("distance", 1.0), 3)
                    st.markdown(
                        f"**[{i}] {entity}** ({etype}) — "
                        f"similarity: `{similarity}`"
                    )
                    preview = src["text"]
                    if len(preview) > 400:
                        preview = preview[:400] + "…"
                    st.text(preview)
                    if i < len(msg["sources"]):
                        st.divider()

# ─── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about a famous person or place…"):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context…"):
            query_type = classify_query(prompt)
            chunks, _  = retrieve(prompt, n_results=n_results)

        try:
            # Stream tokens live so the user sees output immediately
            answer = st.write_stream(
                generate_stream(prompt, chunks, model=model)
            )

            type_label = {
                "person": "👤 person query",
                "place":  "📍 place query",
                "both":   "🔀 mixed / open query",
            }
            st.caption(
                f"{type_label.get(query_type, query_type)} · "
                f"{len(chunks)} chunks retrieved"
            )

            if show_sources and chunks:
                with st.expander("📄 Retrieved source chunks"):
                    for i, src in enumerate(chunks, 1):
                        meta       = src.get("metadata", {})
                        entity     = meta.get("entity", "—")
                        etype      = meta.get("type", "—")
                        similarity = round(1.0 - src.get("distance", 1.0), 3)
                        st.markdown(
                            f"**[{i}] {entity}** ({etype}) — "
                            f"similarity: `{similarity}`"
                        )
                        preview = src["text"]
                        if len(preview) > 400:
                            preview = preview[:400] + "…"
                        st.text(preview)
                        if i < len(chunks):
                            st.divider()

            st.session_state.messages.append({
                "role":       "assistant",
                "content":    answer,
                "sources":    chunks,
                "query_type": query_type,
            })

        except Exception as exc:
            err = (
                f"❌ **Error:** {exc}\n\n"
                "Make sure Ollama is running (`ollama serve`) "
                "and the index is built (`python build_index.py`)."
            )
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err}
            )
