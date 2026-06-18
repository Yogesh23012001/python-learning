Bi-encoder vs cross-encoder
Your retrieval so far is a bi-encoder. The embedder encodes the query into a vector and the chunk into a vector separately, then compares them with cosine similarity. The query and chunk never "see" each other during encoding — they're embedded in isolation and compared after.

Fast: you embed all chunks once at ingestion. At query time you embed just the query and do cheap vector math. This is why it scales to millions of chunks.
Less precise: because query and chunk are encoded separately, the model can't reason about their interaction. It captures "are these about similar topics" but not "does this chunk actually answer this specific query."

A reranker is a cross-encoder. It takes the query and a chunk together as one input and outputs a single relevance score. The model reads them jointly — it can see exactly how the chunk relates to the query.

Precise: the model attends to query and chunk together, judging true relevance, not just topical similarity. It catches "this chunk mentions the topic but doesn't answer the question" — the exact failure you saw in Hour 3.
Slow: you must run the model on every (query, chunk) pair. You can't precompute — the pair didn't exist until query time. This is why you can't rerank your whole corpus — only the handful of candidates first-stage retrieval surfaced.