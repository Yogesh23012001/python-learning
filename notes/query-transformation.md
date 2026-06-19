# Query Transformation

## The mismatch problem (my words)

Questions live in "question-space"; documents live in "answer-space." They can
be about the exact same thing yet embed far apart, because they're *different
kinds of text* — a question is phrased like a question, a doc chunk is phrased
like a statement of fact.

The failure: during query→chunk retrieval I match on semantic similarity of the
two vectors, but the query and the right chunk can land in different regions of
the space purely because of **word/phrasing mismatch** — not because they're
about different things. So the relevant chunk exists, embedding is fine, and
retrieval still misses it because the question doesn't *look* like the answer.

This is the ceiling that hybrid search and reranking can't fully fix: if the
query never lands near the right chunk in the first place, fusion and
cross-encoders are reranking a candidate set that doesn't contain the answer.
Query transformation attacks the problem one step earlier — at the query itself.

## The three techniques (my words)

- **Query expansion** — add related terms to the query so it overlaps more
  vocabulary with the target chunk, nudging its vector closer. Cheap (often no
  LLM, or one small call), but only closes *mild* gaps — it can't bridge a real
  question-space ↔ answer-space divide.
- **Multi-query** — ask an LLM to reword the question N different ways, retrieve
  for each, then fuse the result sets (same RRF idea as hybrid search). Each
  rewording is a different "shot" at landing near the right chunk, so a chunk
  any one of them finds surfaces. Robust, but costs N× the retrieval (and an LLM
  call to generate the rewordings).
- **HyDE** (Hypothetical Document Embeddings) — instead of embedding the
  question, ask an LLM to write a *hypothetical answer* and embed **that**. A
  fake answer is answer-shaped, so it embeds in answer-space, near the real
  answer chunks. Directly bridges the space gap rather than nudging across it.

## Why HyDE works (the key insight, my words)

A fake answer looks more like a real chunk than a question does. A chunk is a
statement; a hypothetical answer is also a statement; a question is not. So the
hypothetical answer embeds *near the real answer chunks* — in the same region of
the space — while the raw question sits off in question-space.

The crucial part: the hypothetical answer **does not need to be correct.** It
just needs to be *answer-shaped*. Even if the LLM hallucinates wrong facts, the
fake answer still uses the vocabulary and sentence-shape of a real answer, which
is exactly what makes it land near the true chunk. It's a **retrieval probe** —
thrown away after retrieval. The actual answer the user sees still comes from
the real chunks I retrieve and ground the generator in; the hypothetical answer
never reaches them.

This ties back to the embeddings lesson: **embeddings match the *kind* of text,
not just the topic.** Two texts on the same topic but of different kinds
(question vs statement) embed apart; two texts of the same kind (fake answer vs
real answer) embed together. HyDE exploits exactly that — it converts a
question into the right *kind* of text before embedding.

## My mismatch test case

The query I'll use in Hour 2 to test whether HyDE actually helps is a
**casual, problem-framed** question whose wording shares almost nothing with the
target chunk:

> "Why does Claude keep forgetting what we talked about earlier in a long chat?"

The right chunk is in the `context-windows` doc, but it talks about *"context
window," "token limits,"* and *"compaction"* — none of the user's words
("forgetting," "talked about earlier," "long chat") appear in it. Pure
question-space vs answer-space gap: same topic, zero lexical overlap, phrased as
a complaint rather than a statement.

Prediction: dense retrieval ranks this chunk low (or misses it). HyDE should
help — a hypothetical answer like *"Claude has a fixed context window measured
in tokens; once a conversation exceeds it, earlier messages fall out unless
compaction summarizes them"* is answer-shaped and full of the chunk's own
vocabulary, so it should embed right next to the real chunk. That's the test:
does the right chunk's rank jump from dense → HyDE on this query.

## HyDE result (Wed H2)
Mismatch query "Claude keeps forgetting in long chat" → context-windows doc.
- Dense: target at rank 1, but only 1 of top-8 relevant (7 strays).
- HyDE: 7 of top-8 from context-windows. Same rank-1, but the WHOLE top-k
  became relevant.
Why: the hypothetical probe used the docs' vocabulary ("context window limit",
"tokens", "exceeded") that the question lacked — landed in the right cluster.
Lesson: "rank of first relevant" showed NO change; "fraction of top-k relevant"
(= Context Precision) showed a big gain. The metric you pick decides what you
see. One designed-favorable query though — Thursday's 50-Q eval is the real test.



