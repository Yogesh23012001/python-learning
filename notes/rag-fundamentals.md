# RAG Fundamentals

## Why RAG exists (3 problems, my words)
[Hallucination, knowledge cutoff, private data. One sentence each on how
RAG addresses it. Connect to Week 2's "give it tools that return facts."]
Rag addresses three problems 
 Hallucination : anything out its confidenece level will answered hallucinated , Rag help to give actual souce text answer using this and site it. so claims can be guranted 

 Knowledge cutoff : knwledge will be in our corpus not the models weight

 private data : helps model to answer question  about data it never have seen  

## The naive pipeline (my words)
[Ingestion: documents → chunk → embed → store.
Query: question → embed → retrieve top-k → LLM answers from chunks.
Write it so you could explain it to another engineer cold.]
We first put our documnet in our vector databse , which involves storing it by breaking it in chunks bcz of larger size , while answering any problem we first embed the question and retrive top k chunk relavant to that problem and feet to LLm it will answer us using them 
## The three things that bite
[1. Same embedder both phases — why.
 2. Retrieval = cosine similarity at scale — connect to Saturday's work.
 3. LLM never learns the corpus — reads at query time.]
 Imagine GPS coordinates. Delhi:(28.6, 77.2) Now imagine your query coordinates are measured in: feet, while your documents are measured in:kilometers, Distances stop making sense.Embedding models create their own semantic universe.If documents are embedded with: MiniLM then queries must also be embedded with: MiniLM because both need to live in the same vector space.That's why:Document Embedding ,Question Embedding must use the same model.

## What makes RAG "advanced" (the failure-mode → fix map)
[The table from Part C, in your words. This is the week's roadmap — each
technique fixes a specific naive-RAG failure.]
Misses keywords?
    → Hybrid Search

Gets okay chunks?
    → Reranking

Question wording differs?
    → Query Transformation

Chunks lose context?
    → Contextual Retrieval

Don't know quality?
    → Ragas

Can't trust answers?
    → Citations & Groundedness

## How I'll know it's working
[Ragas metrics: faithfulness, context precision. The capstone targets:
faithfulness > 0.85, context precision > 0.7. Measurement-driven, like
the Week 2 evals.]

RAG pipeline:

Question
   ↓
Retrieval
   ↓
LLM
   ↓
Answer

Metrics:

Context Precision → Retrieval quality
Faithfulness     → Answer quality

Because RAG can fail in two different places:

Failure 1: Retrieval
Wrong chunks retrieved

Caught by:

Context Precision
Failure 2: Generation
Correct chunks retrieved
but model invents facts

Caught by:

Faithfulness


## Hybrid search nuance (Tue H4)
BM25 found the literal `cache_control: ephemeral` chunk that dense missed
entirely — exact-term win, proven. BUT naive RRF demoted it out of the hybrid
top-5 because it rewards cross-retriever agreement and dense never ranked it.
Open question for Thursday eval: does hybrid actually beat dense on the test
set? If exact-term queries regress, tune fusion (weighted RRF or query routing).