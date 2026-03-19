# Report: `prepare_beir_data.py`

This document provides a detailed explanation of the `prepare_beir_data.py` script located in the `zz_coderuns/baseline1` directory.

## 1. Overview
The primary purpose of this script is to act as a data preparation pipeline for experiments in the `fed-rag` project. It loads standard Information Retrieval (IR) benchmarking datasets from the **BEIR** (Benchmarking IR) suite—specifically targeting datasets like `NFCorpus` (Medical literature) and `SciFact` (Scientific claims). 

It transforms this raw data into a format suitable for **Federated Retrieval-Augmented Generation (fed-rag)** experiments. This implies generating knowledge stores (document repositories) and building train/evaluation pairs for training large language models or retrievers. The comments note that this module is shared by three baseline scripts (`centralized_lsr.py`, `federated_iid_lsr.py`, `federated_noniid_lsr.py`).

## 2. Key Configurations
A section near the top defines global constants that control the data scale and parameters:
*   **`RETRIEVER_MODEL`** (`"sentence-transformers/all-MiniLM-L6-v2"`): The underlying lightweight transformer model used to embed documents and queries.
*   **`MAX_CORPUS_DOCS`** (`1000`): Limits the size of the document knowledge store. This keeps the retrieval task tractable and computationally manageable for testing/baselines.
*   **`MAX_TRAIN_PAIRS`** (`500`) & **`MAX_EVAL_PAIRS`** (`100`): Defines the upper bounds for the number of query-document pairs used for training and testing, respectively.
*   **`TOP_K`** (`10`): The standard number of documents retrieved for a single query.
*   **`MAX_RESPONSE_CHARS`** (`500`): Truncates long document texts to ensure they fit within the input token limits of generation models (like `distilgpt2` with its 1024 token limit).

## 3. Core Functions Breakdown

### `load_beir_dataset(dataset_name)`
Loads the corresponding BEIR dataset from the Hugging Face `datasets` hub. A BEIR dataset typically consists of three parts:
1.  **Corpus**: The collection of all documents.
2.  **Queries**: The questions or claims.
3.  **Qrels**: The relevance judgments (mapping which queries are answered by which documents, with a relevance score).
*   **Fallback logic**: The code contains specific error handling (via `try...except`) to account for differences in how Hugging Face structures different BEIR datasets (e.g., `SciFact` requires building queries dynamically from the corpus titles).
*   **Output**: Returns three objects: `doc_lookup` (dict of doc-id to text), `query_lookup` (dict of query-id to text), and the raw `qrels_ds` dataset.

### `build_train_eval_pairs(doc_lookup, query_lookup, qrels_ds, ...)`
Iterates through the relevance judgments (`qrels_ds`) and constructs matched query-document pairs where the `score > 0` (indicating relevance).
*   It looks up the actual text string for IDs and truncates the document texts.
*   It ensures a matched pair is only created if both the query and the document exist in the loaded lookups.
*   It shuffles the collected pairs (using a fixed seed for reproducibility) and splits them into training and evaluation sets according to the configured maximums.

### `build_knowledge_store(doc_lookup, retriever, max_docs)`
Simulates the actual "database" or vector store of documents.
*   It iterates through the corpus (up to `MAX_CORPUS_DOCS`), truncating them for strict encoding speed to 512 characters.
*   It calculates vector embeddings for each document using the `retriever.encode_context()` function.
*   It wraps each document and its embedding into a `KnowledgeNode` and loads them into an `InMemoryKnowledgeStore`.

### `create_retriever()`
A simple factory function that returns a new instance of the Hugging Face sentence transformer retriever using the configured `RETRIEVER_MODEL`.

### `evaluate_retriever(retriever, knowledge_store, eval_pairs, top_k)`
Performs an evaluation of the retriever's capability to find the relevant document without fine-tuning. For every pair in the evaluation set:
1.  The query is embedded.
2.  The `knowledge_store` is searched for the top K closest documents.
3.  Metrics are accumulated:
    *   **MRR (Mean Reciprocal Rank)**: Evaluates how high the first relevant document is in the ranked list.
    *   **Recall@K**: A binary flag indicating if the relevant document appeared anywhere in the Top K.
    *   **NDCG@K (Normalized Discounted Cumulative Gain)**: Accounts for the rank position of the relevant document, providing a smoother penalty for lower-ranked correct retrievals.

### `setup_dataset(...)` and `filter_eval_pairs_by_store(...)`
The overarching coordinator function (`setup_dataset`).
*   It calls the loaders, builds the stores, and importantly, uses `filter_eval_pairs_by_store` to guarantee that any evaluation pairs generated *only* refer to documents that actually made it into the `knowledge_store` (since the corpus was artificially truncated to `MAX_CORPUS_DOCS`).
*   It packages the final training pairs into a Hugging Face `Dataset` object for downstream model training APIs and returns a dictionary with all the necessary components for a baseline run.

## 4. Main Execution (`if __name__ == "__main__":`)
When run as a standalone script, it executes a "Quick test".
*   It calls `setup_dataset("nfcorpus")` with very small bounds (max 50 docs, max 20 train, 5 eval).
*   It runs the `evaluate_retriever` function to calculate baseline (pre-training) metrics for `MRR`, `Recall@K`, and `NDCG@K` to ensure the retrieval pipeline functions end-to-end.
