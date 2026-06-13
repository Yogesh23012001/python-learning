# Embedding Model Choice for the RAG Pipeline

## The comparison (my numbers)

| model                  | dim | load_s | embed_ms | P@1 | MRR |
|------------------------|-----|--------|----------|-----|-----|
| all-MiniLM-L6-v2       | 384 |  5.75  |  2280.7  | 1.0 | 1.0 |
| BAAI/bge-small-en-v1.5 | 384 | 18.02  |   259.2  | 1.0 | 1.0 |
| BAAI/bge-base-en-v1.5  | 768 | 32.54  |   234.1  | 1.0 | 1.0 |

All three scored **P@1 = 1.0 and MRR = 1.0** — perfect retrieval. That's the
first thing to be honest about: my test set (6 chunks) is too small/easy to
separate the models on *quality*. Nobody "won" on P@1. So the real decision
here comes down to **speed and storage**, not retrieval accuracy — I'd need a
bigger, harder eval set before I could claim one model retrieves better.

The one sharp difference: **MiniLM was ~9x slower to embed** (2280 ms vs
~250 ms for both BGE models). The two BGE models were neck-and-neck on embed
speed despite bge-base being the larger (768-dim, 438 MB) model.

## The three axes, what I saw

- **Dimension**: bge-base is 768-dim, the other two are 384. On *this* test
  set the extra dimensions bought nothing — P@1/MRR were already maxed at 384.
  Bigger dim = 2x the storage and 2x the vector-math at query time for no
  measured quality gain. Bigger isn't better until the eval is hard enough to
  show it.
- **Speed**: MiniLM at 2280 ms for 6 chunks is the outlier — that scales
  *linearly*, so ~380 ms/chunk. On a 100k-chunk corpus that's ~10 hours of
  ingestion vs ~40 min for the BGE models (~43 ms/chunk). At query time the
  same gap is per-question latency the user feels. The BGE pair is the clear
  win on the speed axis.
- **Quality**: tied at 1.0 across the board, so I can't observe the
  "trained-for-retrieval" BGE-beats-MiniLM effect here — the eval doesn't
  discriminate. BGE *is* the retrieval-tuned family, so I'd expect it to pull
  ahead on a harder set, but I'm not going to claim what my numbers don't show.

## My pick and why

**BAAI/bge-small-en-v1.5.** For a RAG pipeline it wins on the axes that
actually moved:

- **Ingestion speed**: 259 ms/6-chunks, ~9x faster than MiniLM. Embedding the
  whole corpus once is the dominant cost, and this is at the fast end.
- **Query latency**: same fast embedder runs on every question — sub-300 ms.
- **Retrieval quality**: tied at P@1 = 1.0, and it's from the retrieval-tuned
  BGE family, so it's the safe bet if/when the eval gets harder.
- **Storage**: 384-dim — *half* the vector size of bge-base for identical
  measured quality. Cheaper index, faster ANN search.

The tradeoff I'm accepting: bge-base's 768 dims *might* retrieve better on a
harder corpus, and I'm giving that up for half the storage and equal scores
today. The slightly longer load_s (18 s vs MiniLM's 5.75 s) is a one-time
cold-start at service boot — irrelevant next to per-chunk embed cost.

## When I'd choose differently

- **Huge corpus, storage-bound** → already on the small (384-dim) model;
  if even that's too much, quantize the vectors or look at a 256-dim model.
- **Quality is critical and budget allows** → switch to an API embedder
  (OpenAI text-embedding-3-large, Voyage) and re-run this same eval on a
  *harder* test set to justify the spend.
- **Specialized domain** (legal, medical, code) → a domain-tuned or
  fine-tuned embedding model, since general-purpose BGE may miss jargon
  similarity.
- **Need to actually rank these three** → build a bigger, adversarial eval
  set first. Today's 1.0/1.0 means the benchmark, not the model, is the
  limiting factor.

## The cost angle (connect to the gateway)

API embedders (OpenAI, Voyage) bill **per token** — the same cost model as the
LLM calls I metered in the gateway. Local embedders like BGE cost **compute,
not dollars**: you pay in load time + embed_ms, not per-request fees. For a
high-ingestion RAG pipeline embedding thousands of chunks, a per-token API
embedder is exactly the kind of spend that needs the same cost-governance /
metering thinking I applied at the gateway — whereas a local BGE model turns
that recurring per-token cost into fixed compute. That's a real argument for
keeping the embedder local unless quality forces the API.
