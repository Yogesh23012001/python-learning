# Vector Databases + Indexes

## The problem (my words)

The semantic cache I built Saturday does a **brute-force similarity search**:
to find the nearest vector it compares the query against *every* stored vector,
one by one. That's **O(N)** — fine at a few hundred entries, but every new item
makes every future lookup slower. I flagged this at the time as the scale gap,
and this is exactly it: linear scan doesn't survive growth. Once the store is
tens of thousands of vectors, each query is doing tens of thousands of cosine
comparisons.

The fix is a **vector database backed by an ANN (approximate nearest neighbor)
index**. Instead of scanning everything, it builds a data structure once that
lets a query jump to the likely-nearest vectors and skip the rest — turning an
O(N) scan into something closer to O(log N).

## Exact vs approximate (my words)

- **Exact NN** = check everything. Guaranteed to return the true nearest
  vectors, but it's the O(N) brute-force scan — correct and slow.
- **ANN** = check clever subsets. The index pre-organizes the vectors (a graph
  or clusters) so a query only explores the regions likely to hold the answer.
  Approximate — it can occasionally miss the true #1 — but dramatically faster.

The trade is **a tiny accuracy loss for a huge speed gain**. For RAG that's a
good deal: I'm already retrieving the top-k chunks and feeding *all* of them to
the LLM, so if ANN swaps the #4 result for the "true" #4 once in a while, the
generated answer is unaffected. I'm not doing anything that needs provably
exact nearest neighbors — I need *good enough, fast*. That's the whole reason
ANN exists and why it's the right default here.

## HNSW vs IVF (my words)

- **HNSW** (Hierarchical Navigable Small World) = a navigable **graph**. Think
  express lanes at the top for big jumps across the space, then local hops at
  the bottom to home in. Result: **fast queries + high recall**, at the cost of
  **more memory** (the graph links) and slower index build.
- **IVF** (Inverted File) = **cluster** the vectors into buckets, and at query
  time only probe the few nearest buckets. **Less memory + faster to build**,
  but **lower recall** — if the answer sits just outside a probed bucket, you
  miss it (unless you probe more buckets, which costs speed).

**My pick: HNSW.** RAG is **read-heavy** — I build the index once at ingestion
and then query it constantly. So I happily pay HNSW's higher build cost and
memory to get the query speed and recall that every user question benefits
from. IVF's cheaper build would matter more for a write-heavy / frequently
re-indexed workload, which this isn't.

## DB choice: why pgvector

The framework I'm using: pick the index/DB that matches my **scale**, my
**existing infra**, and how much **operational complexity** I want to take on.

**Why pgvector for me:**
- I'm **already on Postgres** (it's what I used for cost tracking in the
  gateway) — no new database, no new service to run or learn.
- **Moderate scale** — pgvector with an HNSW index comfortably handles this
  corpus size; I don't need a dedicated vector engine yet.
- **One DB for chunks + vectors + metadata.** The chunk text, its embedding,
  and its metadata live in the same row. SQL `JOIN`s, `WHERE` filters, and
  **transactions** just work — I can filter by metadata and do vector search in
  one query, with real ACID guarantees.
- **No new infra** to deploy, secure, monitor, or pay for.

**When I'd choose differently:** at **billion-vector scale**, or when vector
search is the product's hot path and Postgres becomes the bottleneck, I'd reach
for a purpose-built vector DB — **Qdrant** (self-hosted, fast, good filtering)
or **Pinecone** (managed, hands-off scaling). The line is roughly: pgvector
until the vector workload outgrows what a general-purpose DB can serve, then a
dedicated engine.

## Connection to the gateway

The **same Postgres** I stood up for cost tracking in the gateway now *also*
holds the vectors and the chunk metadata. **One database, two jobs.** That's
the same instinct that made the gateway design feel right — **fewer moving
parts**. Every additional service is another thing to deploy, back up, secure,
and debug at 2am; folding vectors into the Postgres I already operate means the
operational-simplicity argument that justified pgvector is the exact same
"minimize moving parts" reasoning I applied to the gateway. Reuse the boring,
proven infrastructure until scale genuinely forces a specialized tool.
