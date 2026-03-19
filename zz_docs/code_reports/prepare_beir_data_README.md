# `prepare_beir_data.py` Walkthrough

## Purpose
This file is the data and evaluation utility layer for the `quality_aware_fedrag` experiment. It does four jobs:

1. Load a BEIR dataset such as `nfcorpus`.
2. Convert qrels into `(query, response, doc_id)` training/evaluation pairs.
3. Build the retriever-backed in-memory knowledge store.
4. Evaluate retrieval quality with `MRR`, `Recall@k`, and `NDCG@k`.

If you want to understand where the experiment data comes from, this is the first file to read.

## Big Picture Flow
The normal path is:

1. `setup_dataset(...)`
2. `load_beir_dataset(...)`
3. `build_train_eval_pairs(...)`
4. `create_retriever(...)`
5. `build_knowledge_store(...)`
6. `filter_eval_pairs_by_store(...)`
7. Return a bundle used by `federated_noisy_qa.py`

## Important Constants
- `RETRIEVER_MODEL`: SentenceTransformer backbone.
- `MAX_CORPUS_DOCS`: cap for knowledge-store size.
- `MAX_TRAIN_PAIRS`, `MAX_EVAL_PAIRS`: dataset limits.
- `TOP_K`: retrieval depth used during evaluation.
- `SEED`: default seed used when callers do not override it.
- `MAX_RESPONSE_CHARS`: truncates document text before it becomes a training response.

## Main Functions

### `load_beir_dataset(dataset_name="nfcorpus")`
Loads:

- corpus documents
- query texts
- qrels

It builds:

- `doc_lookup`: `doc_id -> doc text`
- `query_lookup`: `query_id -> query text`

There are fallback branches because BEIR splits are not always laid out exactly the same way for every dataset.

### `build_train_eval_pairs(...)`
This is where qrels become usable pairs for the QA-FedAvg experiment.

Each positive qrel becomes a dictionary like:

- `query`
- `response`
- `query_id`
- `doc_id`

Important detail:

- `response` is the truncated relevant document text, not a generated answer.

That matters because the LSR pipeline later uses this text as the teacher-side supervision context.

The function now accepts a `seed` argument, so different experiment seeds actually change the train/eval split instead of only changing the noise injection.

### `build_knowledge_store(doc_lookup, retriever, max_docs=...)`
Builds a FedRAG `InMemoryKnowledgeStore`.

For each document:

- truncate text
- embed with `retriever.encode_context(...)`
- create a `KnowledgeNode`
- load all nodes into the store

This is the corpus the retriever searches against during training and evaluation.

### `create_retriever()`
Returns the `HFSentenceTransformerRetriever` instance used across the experiment.

### `evaluate_retriever(retriever, knowledge_store, eval_pairs, top_k=TOP_K)`
Computes:

- `mrr`
- `recall_at_k`
- `ndcg_at_k`

This is the function used for:

- pre-training evaluation
- post-aggregation evaluation after each federated round

### `filter_eval_pairs_by_store(eval_pairs, knowledge_store)`
Drops evaluation pairs whose relevant document is not present in the capped knowledge store.

This is important because the experiment limits the corpus to `MAX_DOCS`. Without filtering, evaluation would unfairly score queries whose true document was never loaded.

### `setup_dataset(...)`
This is the main entry point used by the training script.

It returns a dictionary containing:

- `retriever`
- `knowledge_store`
- `train_dataset`
- `train_pairs`
- `eval_pairs`
- `doc_lookup`

## How This File Connects to the Others
- `federated_noisy_qa.py` calls `setup_dataset(...)` to get the train pairs, eval pairs, and knowledge store.
- `federated_noisy_qa.py` also calls `evaluate_retriever(...)` before training and after every aggregation round.
- `run_alpha_sweep.py` does not use this file directly; it triggers the training script, which uses this file internally.

## What Changed Recently
- The train/eval pair builder now accepts a real `seed`.
- `setup_dataset(...)` passes that seed through.

That change matters for multi-seed experiments because previously the data split stayed fixed even when the experiment seed changed.

## Reading Tips
If you are going through this file side by side with the code, focus on:

1. How qrels become `(query, response)` pairs.
2. Why `doc_id` is preserved for evaluation.
3. Why evaluation pairs are filtered by knowledge-store coverage.
4. Why a configurable seed is necessary for proper benchmark variance analysis.
