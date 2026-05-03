"""Answer generation via a local Ollama model.

The generator:
1. Builds a grounded prompt from the retrieved context chunks.
2. Sends it to Ollama's /api/generate endpoint using only stdlib urllib.
3. Returns the model's response string (blocking) or yields tokens (streaming).

The system prompt instructs the model to answer *only* from the provided
context and return "I don't know" when the answer is absent — this is the
primary anti-hallucination guard.  Temperature is set low (0.1) to make
answers more factual and less creative.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterator

OLLAMA_BASE   = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
TIMEOUT_SEC   = 120

_SYSTEM = (
    "You are a factual assistant that answers questions about famous people and places. "
    "You MUST answer ONLY from the provided context. "
    "If the answer is not contained in the context, respond with exactly: "
    "\"I don't know based on the available information.\" "
    "Do not add facts, dates, or claims not present in the context. "
    "Be concise."
)


def _build_prompt(query: str, chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta   = chunk.get("metadata", {})
        entity = meta.get("entity", "Unknown")
        etype  = meta.get("type", "")
        parts.append(f"[Source {i} — {entity} ({etype})]\n{chunk['text']}")

    context = "\n\n---\n\n".join(parts)

    return (
        f"{_SYSTEM}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def generate(
    query:       str,
    chunks:      list[dict],
    model:       str   = DEFAULT_MODEL,
    temperature: float = 0.1,
) -> str:
    """Blocking generation — returns the full answer string."""
    if not chunks:
        return "I don't know based on the available information."

    prompt  = _build_prompt(query, chunks)
    payload = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()
    except Exception as exc:
        return f"[Generation error: {exc}]"


def generate_stream(
    query:       str,
    chunks:      list[dict],
    model:       str   = DEFAULT_MODEL,
    temperature: float = 0.1,
) -> Iterator[str]:
    """Streaming generation — yields tokens as they arrive."""
    if not chunks:
        yield "I don't know based on the available information."
        return

    prompt  = _build_prompt(query, chunks)
    payload = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  True,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj   = json.loads(line)
                    token = obj.get("response", "")
                    if token:
                        yield token
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        yield f"\n[Generation error: {exc}]"


def check_ollama() -> tuple[bool, list[str]]:
    """Probe Ollama; return (is_running, [model_names])."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data   = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []
