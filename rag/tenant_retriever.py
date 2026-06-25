"""Tenant-scoped retrieval with ABAC.

Isolation is enforced IN the vector query (WHERE tenant_id = %s), before the
LIMIT — so other tenants' chunks are never candidates. This is "defense by
unreachability": the wrong data isn't filtered out, it's never considered.

ABAC adds attribute checks (sensitivity vs. clearance) as additional WHERE
predicates — same principle, more attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from sentence_transformers import SentenceTransformer

from rag.pipeline import RetrievedChunk

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@dataclass(frozen=True)
class Principal:
    """Who is asking — carries the attributes ABAC decisions are made on.

    Mirrors an ABAC subject: identity (tenant) + attributes (clearance).
    """

    tenant_id: str
    clearance: str = "normal"  # "normal" | "confidential"

    def allowed_sensitivities(self) -> list[str]:
        """ABAC rule: what sensitivity levels this principal may see."""
        if self.clearance == "confidential":
            return ["normal", "confidential"]
        return ["normal"]


def tenant_retrieve(
    question: str,
    principal: Principal,
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve ONLY chunks this principal is allowed to see.

    Isolation + ABAC enforced in the WHERE clause, before LIMIT — so
    disallowed chunks are never candidates, not filtered after the fact.
    """
    model = _get_model()
    q_emb = model.encode(question, normalize_embeddings=True)
    q_str = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    allowed = principal.allowed_sensitivities()
    placeholders = ",".join(["%s"] * len(allowed))

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT doc_id, chunk_text, metadata, tenant_id, sensitivity,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            WHERE tenant_id = %s
              AND sensitivity IN ({placeholders})
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q_str, principal.tenant_id, *allowed, q_str, top_k),
        )
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            doc_id=d,
            text=t,
            score=float(s),
            metadata={**(m or {}), "tenant_id": tid, "sensitivity": sens},
        )
        for (d, t, m, tid, sens, s) in rows
    ]


if __name__ == "__main__":
    # Same query, three different principals — see how results differ
    q = "How do I structure prompts and use caching?"

    principals = [
        Principal("tenant_acme"),
        Principal("tenant_globex", clearance="normal"),
        Principal("tenant_globex", clearance="confidential"),
        Principal("public"),
    ]

    for p in principals:
        print(f"\n=== Principal: tenant={p.tenant_id} clearance={p.clearance} ===")
        results = tenant_retrieve(q, p, top_k=3)
        if not results:
            print("  (no accessible chunks)")
        for r in results:
            print(
                f"  [{r.metadata['tenant_id']}/{r.metadata['sensitivity']}] "
                f"{r.doc_id}  sim={r.score:.3f}"
            )
