"""Document loader: fetch web pages, extract clean main-content text.

The first stage of the RAG ingestion pipeline. Fetches each URL, strips HTML
chrome (nav, scripts, sidebars) down to article text, returns structured
documents ready for chunking.

The unglamorous truth of RAG: most ingestion pain is getting clean text out
of messy sources, not the embeddings. This stage earns its keep.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import httpx
import trafilatura


@dataclass(frozen=True)
class LoadedDocument:
    """A fetched, cleaned document ready for chunking."""

    doc_id: str  # stable id (we use the URL slug)
    url: str
    title: str
    text: str  # cleaned main-content text


# The corpus — a coherent spread of Anthropic doc pages across topics
# you've been learning, so you'll know the answers when testing retrieval.
CORPUS_URLS = [
    # Prompt engineering (the consolidated best-practices page is one doc now)
    "https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview",
    "https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices",
    # Distinct feature pages — these are genuinely separate content
    "https://docs.claude.com/en/docs/build-with-claude/tool-use/overview",
    "https://docs.claude.com/en/docs/build-with-claude/prompt-caching",
    "https://docs.claude.com/en/docs/build-with-claude/embeddings",
    "https://docs.claude.com/en/docs/build-with-claude/context-windows",
    "https://docs.claude.com/en/docs/build-with-claude/citations",
    "https://docs.claude.com/en/docs/build-with-claude/extended-thinking",
    "https://docs.claude.com/en/docs/build-with-claude/vision",
    "https://docs.claude.com/en/docs/build-with-claude/structured-outputs",
    "https://docs.claude.com/en/docs/build-with-claude/streaming",
    "https://docs.claude.com/en/docs/build-with-claude/batch-processing",
    "https://docs.claude.com/en/docs/build-with-claude/token-counting",
    "https://docs.claude.com/en/docs/build-with-claude/pdf-support",
    "https://docs.claude.com/en/docs/test-and-evaluate/develop-tests",
]


_BOILERPLATE_MARKERS = [
    "We use cookies to deliver",
    "You can read our Cookie Policy",
]


def _strip_boilerplate(text: str) -> str:
    """Remove known boilerplate lines (cookie banners, etc.)."""
    lines = text.split("\n")
    cleaned = [ln for ln in lines if not any(marker in ln for marker in _BOILERPLATE_MARKERS)]
    return "\n".join(cleaned).strip()


def _slug(url: str) -> str:
    """Stable doc_id from the URL's last path segments."""
    parts = url.rstrip("/").split("/")
    return "/".join(parts[-2:])  # e.g. "prompt-engineering/overview"


def load_url(client: httpx.Client, url: str) -> LoadedDocument | None:
    """Fetch one URL, extract clean text. Returns None on failure."""
    try:
        resp = client.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  FETCH FAILED {url}: {type(exc).__name__}")
        return None

    # trafilatura extracts main content, dropping nav/scripts/boilerplate
    extracted = trafilatura.extract(
        resp.text,
        include_comments=False,
        include_tables=True,
        favor_recall=True,  # err toward keeping content
    )
    if not extracted:
        print(f"  EXTRACT EMPTY/THIN {url}")
        return None

    # Fix 1: drop known boilerplate (cookie banners, etc.) before the size gate
    extracted = _strip_boilerplate(extracted)

    # Fix 3: raise the minimum-length bar — 378-char stubs get dropped
    if len(extracted.strip()) < 500:
        print(f"  EXTRACT EMPTY/THIN {url}")
        return None

    # Title: trafilatura metadata, fallback to slug
    meta = trafilatura.extract_metadata(resp.text)
    title = (meta.title if meta and meta.title else _slug(url)).strip()

    return LoadedDocument(
        doc_id=_slug(url),
        url=url,
        title=title,
        text=extracted.strip(),
    )


def load_corpus(urls: list[str] = CORPUS_URLS) -> list[LoadedDocument]:
    """Fetch and clean all corpus URLs. Polite delay between requests."""
    docs: list[LoadedDocument] = []
    seen_hashes: set[str] = set()
    headers = {"User-Agent": "Mozilla/5.0 (RAG learning project; respectful crawler)"}

    with httpx.Client(headers=headers) as client:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {_slug(url)}...", end=" ", flush=True)
            doc = load_url(client, url)
            if not doc:
                time.sleep(1.0)  # polite delay — don't hammer the server
                continue
            # Fix 2: dedup — skip if we've already seen identical content
            content_hash = hashlib.sha256(doc.text.encode()).hexdigest()
            if content_hash in seen_hashes:
                print("DUPLICATE of earlier doc — skipping")
                time.sleep(1.0)
                continue
            seen_hashes.add(content_hash)
            print(f"ok ({len(doc.text)} chars)")
            docs.append(doc)
            time.sleep(1.0)  # polite delay — don't hammer the server

    print(f"\nLoaded {len(docs)}/{len(urls)} documents.")
    total_chars = sum(len(d.text) for d in docs)
    print(
        f"Total: {total_chars:,} chars (~{total_chars // 4:,} tokens, ~{total_chars // 1500} chunks at 1500 chars)"
    )
    return docs


if __name__ == "__main__":
    docs = load_corpus()
    # Show a preview of each
    print("\n" + "=" * 60)
    for d in docs[:3]:
        print(f"\n[{d.doc_id}] {d.title}")
        print(f"  {d.text[:200]!r}")
