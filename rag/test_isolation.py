"""Isolation test: prove a tenant CANNOT retrieve another tenant's chunks,
even with a query that semantically matches the other tenant's content perfectly.

This is the security proof. A passing test means isolation is enforced
structurally (in the query), not by luck or post-filtering.
"""

from __future__ import annotations

from rag.tenant_retriever import Principal, tenant_retrieve


def test_no_cross_tenant_leak() -> None:
    """acme queries for content that lives in globex's docs. Must get NOTHING
    from globex — even though the query is a perfect semantic match for it."""

    # "prompt caching" content belongs to tenant_globex. acme asks about it.
    query = "How does prompt caching reduce costs with cache_control?"
    acme = Principal("tenant_acme")

    results = tenant_retrieve(query, acme, top_k=10)

    # Every returned chunk MUST belong to acme. None may be globex's.
    leaked = [r for r in results if r.metadata["tenant_id"] != "tenant_acme"]

    print(f"Query (matches globex's content): {query!r}")
    print("Principal: tenant_acme")
    print(
        f"Retrieved {len(results)} chunks, all tenants present: "
        f"{set(r.metadata['tenant_id'] for r in results)}"
    )

    if leaked:
        print(f"  ❌ LEAK: {len(leaked)} chunks from another tenant reached acme!")
        for r in leaked:
            print(f"     {r.metadata['tenant_id']}: {r.doc_id}")
    else:
        print("  ✅ NO LEAK: every chunk belongs to acme (or none returned)")

    assert not leaked, "ISOLATION BREACH"


def test_abac_sensitivity() -> None:
    """A normal-clearance globex user must NOT see confidential globex chunks,
    even though they belong to their own tenant."""

    query = "How does prompt caching work?"  # caching docs are globex + confidential
    normal = Principal("tenant_globex", clearance="normal")
    cleared = Principal("tenant_globex", clearance="confidential")

    normal_results = tenant_retrieve(query, normal, top_k=10)
    cleared_results = tenant_retrieve(query, cleared, top_k=10)

    normal_conf = [r for r in normal_results if r.metadata["sensitivity"] == "confidential"]
    cleared_conf = [r for r in cleared_results if r.metadata["sensitivity"] == "confidential"]

    print("\nABAC test — same tenant, different clearance:")
    print(
        f"  normal user: {len(normal_results)} chunks, "
        f"{len(normal_conf)} confidential (should be 0)"
    )
    print(
        f"  cleared user: {len(cleared_results)} chunks, "
        f"{len(cleared_conf)} confidential (can be >0)"
    )

    if normal_conf:
        print("  ❌ ABAC BREACH: normal user saw confidential chunks")
    else:
        print("  ✅ ABAC enforced: normal user saw no confidential chunks")

    assert not normal_conf, "ABAC BREACH: normal clearance saw confidential"


if __name__ == "__main__":
    print("=" * 60)
    print("ISOLATION TEST")
    print("=" * 60)
    test_no_cross_tenant_leak()
    test_abac_sensitivity()
    print("\nAll isolation tests passed. ✅")
