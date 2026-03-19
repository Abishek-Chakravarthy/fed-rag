# Mid-Semester Review — Execution Roadmap

This document is a step-by-step guide for preparing the mid-sem review demo. It covers dataset selection, baseline metrics, and partial implementation of both objectives.

---

## Phase 1: Dataset Setup (Day 1–2)

### 1.1 Which Dataset?

I recommend **NFCorpus** (Medical/Nutrition) + **SciFact** (Scientific Claims). Both are small, fast to train, and naturally heterogeneous.

| Dataset | Documents | Queries | Domain | Train Queries |
|---------|-----------|---------|--------|---------------|
| **NFCorpus** | 3,633 | 3,237 | Medical / Nutrition | ~2,500 |
| **SciFact** | 5,183 | 1,261 | Scientific Claims | ~900 |

**Why these two?**
- Small enough to train on your M1 Pro in minutes (not hours).
- Naturally different domains → perfect for proving Domain-Aware selection.
- NFCorpus has relevance judgments (qrels) → you can compute MRR/Recall.

### 1.2 How to Load and Adapt

The fed-rag pipeline expects `(query, response)` pairs. BEIR datasets have `(query, relevant_document)` pairs — which is what we need for LSR training. The "response" in LSR is the text of the relevant document.

```python
from datasets import load_dataset

# ── Load NFCorpus ──
corpus = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
queries = load_dataset("BeIR/nfcorpus", "queries")["queries"]
qrels = load_dataset("BeIR/nfcorpus-qrels", split="train")

# Build a lookup: doc_id → doc_text
doc_lookup = {row["_id"]: row["text"] for row in corpus}
query_lookup = {row["_id"]: row["text"] for row in queries}

# Create (query, response) pairs from qrels
# qrels has: query_id, corpus_id, score
train_data = []
for row in qrels:
    if row["score"] > 0:  # only positive relevance
        q_text = query_lookup.get(row["query-id"], None)
        d_text = doc_lookup.get(row["corpus-id"], None)
        if q_text and d_text:
            train_data.append({"query": q_text, "response": d_text})

# Convert to HuggingFace Dataset
from datasets import Dataset
train_dataset = Dataset.from_list(train_data)
print(f"Training pairs: {len(train_dataset)}")
# Expected: ~2000-3000 pairs for NFCorpus
```

### 1.3 Create the Knowledge Store

The corpus documents become the knowledge store (the vector database the retriever searches):

```python
# Index all corpus documents into the knowledge store
from fed_rag.knowledge_stores import InMemoryKnowledgeStore
from fed_rag.data_structures import KnowledgeNode

knowledge_store = InMemoryKnowledgeStore()
nodes = []
for doc in corpus:
    embedding = retriever.encode(doc["text"])  # SentenceTransformer
    node = KnowledgeNode(
        embedding=embedding.tolist(),
        text_content=doc["text"],
        metadata={"doc_id": doc["_id"]},
    )
    nodes.append(node)
knowledge_store.load_nodes(nodes)
```

### 1.4 Create Client Splits

**For Objective 1 (Quality-Aware):** Split NFCorpus into 2 clients:
```python
# Client 0: Clean data (first half)
client_0_data = train_data[:len(train_data)//2]

# Client 1: Noisy data (second half, with 50% shuffled responses)
import random
client_1_data = train_data[len(train_data)//2:]
for i in range(0, len(client_1_data), 2):  # corrupt every other pair
    j = random.randint(0, len(client_1_data)-1)
    client_1_data[i]["response"] = client_1_data[j]["response"]
```

**For Objective 2 (Domain-Aware):** Use different datasets per client:
```python
# Client A: NFCorpus (Medical)
# Client B: SciFact (Science)
# Target domain: Medical → Client A should be selected, Client B down-weighted
```

---

## Phase 2: Baseline Metrics Collection (Day 2–3)

### 2.1 What to Measure

Run **three** configurations and collect the same metrics for each:

| Configuration | Description |
|---------------|-------------|
| **Centralized** | All data pooled, standard LSR training |
| **FedAvg (IID)** | 2 clients, data evenly split, 3-5 rounds |
| **FedAvg (Non-IID)** | 2 clients, different domains, 3-5 rounds |

### 2.2 Metrics to Collect Per Configuration

After each training round (or epoch for centralized), evaluate on the **test set** queries:

```python
def evaluate_retriever(retriever, test_queries, corpus, qrels, k=10):
    """Compute MRR and Recall@k."""
    mrr_total = 0
    recall_total = 0

    for query_id, query_text in test_queries.items():
        # Get relevant doc IDs from qrels
        relevant_docs = {d for d, s in qrels[query_id].items() if s > 0}

        # Retrieve top-k using current retriever
        query_emb = retriever.encode(query_text)
        # ... search knowledge_store for top-k nearest ...
        retrieved_ids = [node.metadata["doc_id"] for node in top_k_results]

        # MRR: rank of first relevant doc
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                mrr_total += 1.0 / rank
                break

        # Recall@k: fraction of relevant docs in top-k
        found = len(set(retrieved_ids) & relevant_docs)
        recall_total += found / len(relevant_docs) if relevant_docs else 0

    n = len(test_queries)
    return {"MRR": mrr_total / n, "Recall@k": recall_total / n}
```

### 2.3 Expected Output for Review

Present a table like this:

| Method | Round | LSR Loss | MRR | Recall@10 |
|--------|-------|----------|-----|-----------|
| Centralized | Epoch 1 | 0.045 | 0.32 | 0.28 |
| Centralized | Epoch 3 | 0.012 | 0.48 | 0.41 |
| FedAvg (IID) | Round 1 | 0.052 | 0.29 | 0.24 |
| FedAvg (IID) | Round 3 | 0.018 | 0.43 | 0.37 |
| FedAvg (Non-IID) | Round 1 | 0.061 | 0.25 | 0.20 |
| FedAvg (Non-IID) | Round 3 | 0.031 | 0.35 | 0.29 |

**Key story to tell:** The "FL Gap" — FedAvg is worse than Centralized, and Non-IID FedAvg is the worst. This is the problem our two ideas solve.

---

## Phase 3: Implement Quality-Aware FedAvg (Day 3–4)

### 3.1 Task Breakdown

| Sub-Task | Effort | What it proves |
|----------|--------|----------------|
| **3.1.1** Create `QualityAwareFedAvg` class | 1-2 hrs | Core implementation |
| **3.1.2** Run with clean vs noisy clients | 1 hr | Quality detection works |
| **3.1.3** Collect metrics and compare | 1 hr | Show improvement over baseline |

### 3.2 What to Show in Mid-Sem

1. **Code:** The `QualityAwareFedAvg` strategy class (it's ~60 lines of code).
2. **Result:** A comparison table showing QA-FedAvg outperforms standard FedAvg when a noisy client is present.
3. **Plot:** Per-round loss curves for FedAvg vs QA-FedAvg (the QA line should be smoother and lower).

---

## Phase 4: Implement Domain-Aware Selection (Day 4–5)

### 4.1 Task Breakdown

| Sub-Task | Effort | What it proves |
|----------|--------|----------------|
| **4.1.1** Compute domain embeddings per client | 30 min | Domain representation works |
| **4.1.2** Compute and display domain similarity matrix | 30 min | Visual proof of heterogeneity |
| **4.1.3** Create `DomainAwareFedAvg` class | 1-2 hrs | Selection mechanism works |
| **4.1.4** Run with multi-domain clients | 1 hr | Show improvement |

### 4.2 What to Show in Mid-Sem

1. **Domain Similarity Matrix:** A heatmap showing cosine similarity between client domain centroids. This visually proves that NFCorpus (Medical) and SciFact (Science) are far apart in embedding space.

```python
import numpy as np

# Domain centroids (already computed)
centroids = {
    "NFCorpus (Medical)": centroid_nfcorpus,   # shape: (384,)
    "SciFact (Science)": centroid_scifact,       # shape: (384,)
}

# Cosine similarity matrix
labels = list(centroids.keys())
matrix = np.zeros((len(labels), len(labels)))
for i, l1 in enumerate(labels):
    for j, l2 in enumerate(labels):
        c1, c2 = centroids[l1], centroids[l2]
        matrix[i][j] = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))

# Plot heatmap using matplotlib/seaborn
```

2. **Selection Log:** Show which clients were selected per round (e.g., "Round 1: Selected Client A (medical, d=0.87), Excluded Client B (science, d=0.23)").
3. **Result:** Medical MRR is higher with DAS-FedAvg than with full-participation FedAvg.

---

## Summary: What to Present

| Slide | Content |
|-------|---------|
| **Problem Statement** | FedAvg ignores quality + domain → degraded RAG performance |
| **System Architecture** | RAG components + Flower FL flow diagram |
| **Baseline Demo** | Centralized vs FedAvg metrics table + loss curves |
| **The FL Gap** | Show Non-IID FedAvg is worse than Centralized |
| **Idea 1: QA-FedAvg** | Theory + code + metrics showing improvement |
| **Idea 2: DAS-FedAvg** | Domain heatmap + selection logs + metrics showing improvement |
| **Future Work** | Full evaluation, dynamic α/τ tuning, larger datasets |
