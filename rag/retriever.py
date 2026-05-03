"""Query classifier and retriever.

Classification strategy: keyword-based entity matching
------------------------------------------------------
We maintain exhaustive lower-case name sets for all 40 ingested entities.
On each query we scan for any of those names.

Routing logic:
  Person names found, no place names  →  filter type=person
  Place names found, no person names  →  filter type=place
  Both found, or neither found        →  no filter (search full collection)

"Both / neither" is the fallback that handles:
  * comparison questions  ("Compare Einstein and Eiffel Tower")
  * thematic questions    ("Which person is associated with electricity?")
  * out-of-domain queries ("Who is the president of Mars?")

This is intentionally simple — rule-based routing avoids adding an LLM
call just for classification, keeping latency low.
"""
from __future__ import annotations

from rag.embedder    import embed_one
from rag.vector_store import get_collection, query_store

# ─── Known entity name sets (lower-case) ────────────────────────────────────

PERSON_NAMES: frozenset[str] = frozenset({
    "albert einstein", "einstein",
    "marie curie", "curie",
    "leonardo da vinci", "da vinci", "leonardo",
    "william shakespeare", "shakespeare",
    "ada lovelace", "lovelace",
    "nikola tesla", "tesla",
    "lionel messi", "messi",
    "cristiano ronaldo", "ronaldo",
    "taylor swift", "swift",
    "frida kahlo", "kahlo",
    "isaac newton", "newton",
    "charles darwin", "darwin",
    "mahatma gandhi", "gandhi",
    "napoleon bonaparte", "napoleon",
    "cleopatra",
    "stephen hawking", "hawking",
    "ludwig van beethoven", "beethoven",
    "wolfgang amadeus mozart", "mozart",
    "pablo picasso", "picasso",
    "abraham lincoln", "lincoln",
})

PLACE_NAMES: frozenset[str] = frozenset({
    "eiffel tower", "eiffel",
    "great wall of china", "great wall",
    "taj mahal",
    "grand canyon",
    "machu picchu",
    "colosseum",
    "hagia sophia",
    "statue of liberty",
    "pyramids of giza", "pyramids", "giza",
    "mount everest", "everest",
    "stonehenge",
    "acropolis of athens", "acropolis",
    "angkor wat", "angkor",
    "chichen itza",
    "vatican city", "vatican",
    "niagara falls", "niagara",
    "amazon river", "amazon",
    "sahara desert", "sahara",
    "great barrier reef", "barrier reef",
    "yellowstone national park", "yellowstone",
})


def classify_query(query_text: str) -> str:
    """Return 'person', 'place', or 'both'.

    'both' signals the retriever to search the full collection with no filter.
    """
    q          = query_text.lower()
    has_person = any(name in q for name in PERSON_NAMES)
    has_place  = any(name in q for name in PLACE_NAMES)

    if has_person and not has_place:
        return "person"
    if has_place and not has_person:
        return "place"
    return "both"


def retrieve(
    query_text: str,
    n_results:  int = 5,
    force_type: str | None = None,
) -> tuple[list[dict], str]:
    """Classify the query, then retrieve the top-k most relevant chunks.

    Args:
        query_text: The user's natural-language question.
        n_results:  Number of chunks to return.
        force_type: Override automatic classification ('person'/'place'/'both').

    Returns:
        (results, query_type) where results is a list of dicts with keys
        'text', 'metadata', and 'distance'.
    """
    query_type = force_type or classify_query(query_text)
    embedding  = embed_one(query_text)
    collection = get_collection()

    where: dict | None = None
    if query_type in ("person", "place"):
        where = {"type": {"$eq": query_type}}

    results = query_store(collection, embedding, n_results=n_results, where=where)

    # Fallback: if the filtered search returns too few results, widen to all
    if len(results) < 2 and where is not None:
        results = query_store(collection, embedding, n_results=n_results, where=None)

    return results, query_type
