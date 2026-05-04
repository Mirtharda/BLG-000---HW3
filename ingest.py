"""Wikipedia ingestion script.

Fetches full Wikipedia article text for 20 famous people and 20 famous
places using the Wikipedia API (no third-party library — plain urllib).
Raw content is stored in a local SQLite database at data/wikipedia.db.

Usage
-----
    python ingest.py            # ingest all, skip already-fetched entries
    python ingest.py --reset    # drop DB and re-ingest everything
    python ingest.py --list     # print what is currently stored

The script is intentionally rate-limited (0.5 s between requests) to be
polite to the Wikimedia servers.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ─── Entity lists ────────────────────────────────────────────────────────────

PEOPLE: list[str] = [
    # Required by assignment spec
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Additional (total ≥ 20)
    "Isaac Newton",
    "Charles Darwin",
    "Mahatma Gandhi",
    "Napoleon",
    "Cleopatra",
    "Stephen Hawking",
    "Ludwig van Beethoven",
    "Wolfgang Amadeus Mozart",
    "Pablo Picasso",
    "Abraham Lincoln",
]

PLACES: list[str] = [
    # Required by assignment spec
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",
    "Mount Everest",
    # Additional (total ≥ 20)
    "Stonehenge",
    "Acropolis of Athens",
    "Angkor Wat",
    "Chichen Itza",
    "Vatican City",
    "Niagara Falls",
    "Amazon River",
    "Sahara",
    "Great Barrier Reef",
    "Yellowstone National Park",
]

DB_PATH  = Path("data/wikipedia.db")
WIKI_API = "https://en.wikipedia.org/w/api.php"
DELAY    = 0.5   # seconds between requests


# ─── Database helpers ─────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT    NOT NULL UNIQUE,
            type       TEXT    NOT NULL CHECK(type IN ('person', 'place')),
            content    TEXT    NOT NULL,
            url        TEXT,
            char_count INTEGER,
            fetched_at TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


# ─── Wikipedia API fetcher ────────────────────────────────────────────────────

def fetch_wikipedia(title: str) -> tuple[str, str]:
    """Return (plaintext_content, page_url) for a Wikipedia article.

    Uses the MediaWiki action=query API with prop=extracts to get the full
    article as plain text (no wiki markup).  No third-party library used.
    """
    params = urllib.parse.urlencode({
        "action":         "query",
        "prop":           "extracts|info",
        "titles":         title,
        "format":         "json",
        "explaintext":    "true",       # plain text, no HTML/markup
        "exsectionformat": "plain",     # section headers as plain text
        "inprop":         "url",        # include canonical URL
    })
    api_url = f"{WIKI_API}?{params}"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "BLG000-HW3/1.0 (educational project)"},
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pages = data.get("query", {}).get("pages", {})
    page  = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title!r}")

    content = page.get("extract", "").strip()
    url     = page.get("fullurl",
                        f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}")

    if not content:
        raise ValueError(f"Empty extract returned for: {title!r}")

    return content, url


# ─── Ingestion loop ───────────────────────────────────────────────────────────

def ingest_entities(
    conn:        sqlite3.Connection,
    entities:    list[str],
    entity_type: str,
) -> None:
    for title in entities:
        existing = conn.execute(
            "SELECT id FROM documents WHERE title = ?", (title,)
        ).fetchone()
        if existing:
            print(f"  [skip]  {title}  (already in DB)")
            continue

        for attempt in range(3):
            try:
                print(f"  [fetch] {title}…", end=" ", flush=True)
                content, url = fetch_wikipedia(title)
                conn.execute(
                    "INSERT INTO documents (title, type, content, url, char_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (title, entity_type, content, url, len(content)),
                )
                conn.commit()
                print(f"✓  {len(content):,} chars")
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f"✗  rate limited, retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    print(f"✗  ERROR: {exc}")
                    break

        time.sleep(DELAY)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia pages for the RAG project."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop the database and re-ingest everything from scratch.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print a summary of what is currently stored and exit.",
    )
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    if args.reset:
        conn.execute("DROP TABLE IF EXISTS documents")
        print("[reset] database cleared\n")

    init_db(conn)

    if args.list:
        rows = conn.execute(
            "SELECT title, type, char_count, fetched_at FROM documents ORDER BY type, title"
        ).fetchall()
        if not rows:
            print("Database is empty. Run  python ingest.py  to ingest data.")
        else:
            print(f"{'Title':<35} {'Type':<8} {'Chars':>8}  Fetched")
            print("-" * 65)
            for title, etype, chars, fetched in rows:
                print(f"{title:<35} {etype:<8} {chars or 0:>8,}  {fetched}")
        conn.close()
        return

    print(f"Ingesting {len(PEOPLE)} people…")
    ingest_entities(conn, PEOPLE, "person")

    print(f"\nIngesting {len(PLACES)} places…")
    ingest_entities(conn, PLACES, "place")

    # Summary
    print("\n=== Ingestion complete ===")
    for row in conn.execute(
        "SELECT type, COUNT(*), SUM(char_count) FROM documents GROUP BY type"
    ):
        print(f"  {row[0]}s: {row[1]} documents, {row[2]:,} total chars")

    conn.close()


if __name__ == "__main__":
    main()
