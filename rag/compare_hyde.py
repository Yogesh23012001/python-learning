"""Compare dense vs HyDE retrieval on a mismatch query.

The test of the Hour 1 prediction: does the right chunk's rank jump when we
retrieve via a hypothetical answer instead of the raw question?
"""

from __future__ import annotations

import sys

from rag.hyde import hyde_retrieve
from rag.retriever import retrieve

q = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Why does Claude keep forgetting what we talked about earlier in a long chat?"
)
target_doc = sys.argv[2] if len(sys.argv) > 2 else "context-windows"

print(f"Query: {q!r}")
print(f"Target doc (where the answer should be): {target_doc!r}\n")


def show(label: str, chunks: list) -> None:
    print(f"--- {label} ---")
    target_rank = None
    for i, rc in enumerate(chunks, 1):
        hit = " <-- TARGET" if target_doc in rc.doc_id else ""
        if target_doc in rc.doc_id and target_rank is None:
            target_rank = i
        print(f"  {i}. [{rc.doc_id}] {rc.score:.3f}{hit}")
    print(f"  >> target doc first appears at rank: {target_rank or 'NOT in top-k'}\n")


show("DENSE (embed the question)", retrieve(q, top_k=8))
show("HyDE (embed a hypothetical answer)", hyde_retrieve(q, top_k=8))
