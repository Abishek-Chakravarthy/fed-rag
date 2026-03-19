# Code Contribution Report

## Overview

All code was written from scratch on top of the open-source **FedRAG** library and **Flower** FL framework. There are two directories:

| Directory | Purpose | Files |
|-----------|---------|-------|
| `zz_coderuns/baseline/` | Baseline experiments (Centralized, Fed-IID, Fed-NonIID) | 4 Python scripts, 3 CSVs |
| `zz_coderuns/quality_aware_fedrag/` | QA-FedAvg implementation + experiments | 4 Python scripts, 5 CSVs |

---

## Shared Utility: `prepare_beir_data.py` (276 lines)

This script is the **data backbone** shared by all experiments. It exists in both directories (with minor differences for non-IID support in the baseline version).

### What it does:
1. **`load_beir_dataset()`** — Downloads BEIR datasets (NFCorpus, SciFact) from HuggingFace. Parses the corpus, queries, and relevance judgments (qrels) into Python dicts.
2. **`build_train_eval_pairs()`** — Converts raw qrels into `(query, response)` training pairs where the "response" is the text of a positively relevant document (truncated to 500 chars to fit within distilgpt2's 1024-token limit). Splits into train and eval sets.
3. **`build_knowledge_store()`** — Encodes up to 1000 corpus documents using the retriever's context encoder and stores them as `KnowledgeNode` objects in FedRAG's `InMemoryKnowledgeStore`.
4. **`evaluate_retriever()`** — Computes **MRR**, **Recall@k**, and **NDCG@k** on held-out eval pairs by encoding each query, retrieving top-k from the store, and comparing against ground-truth relevant doc IDs.
5. **`setup_dataset()`** — One-call convenience that chains all of the above steps.
6. **`filter_eval_pairs_by_store()`** — Filters eval pairs to only keep those whose relevant doc exists in the (possibly subsampled) knowledge store.

### Why I built this:
The FedRAG library has no built-in data loading for standard IR benchmarks. I needed a clean, reusable pipeline that all experiments (centralized, federated, QA) could share.

---

## Baseline Scripts (3 scripts)

### 1. `centralized_lsr.py` (198 lines) — Upper Bound

**Purpose:** Train the retriever on ALL data pooled together (no federation). This is the performance ceiling.

**How it works:**
- Loads all 250 training pairs from NFCorpus
- For each of 3 epochs:
  - Resets `AcceleratorState` (required workaround for SentenceTransformer re-initialization)
  - Builds a fresh `RAGSystem` (retriever + frozen generator + knowledge store)
  - Creates `HuggingFaceTrainerForLSR` with `SentenceTransformerTrainingArguments`
  - Trains for 1 epoch via `HuggingFaceRAGTrainerManager.train()`
  - Rebuilds knowledge store with updated encoder weights for fair evaluation
  - Evaluates MRR, Recall@10, NDCG@10
  - Logs loss, metrics, and weight hash to `centralized_results.csv`

**Key design decision:** Training epoch-by-epoch (not 3 epochs at once) so we can evaluate and log metrics after each epoch.

---

### 2. `federated_iid_lsr.py` (318 lines) — Fed-IID Baseline

**Purpose:** Standard FedAvg with IID data distribution. Shows the "FL gap" vs centralized.

**How it works:**
- `split_iid()` — Randomly shuffles 500 training pairs and splits them equally into 3 client splits of ~167 pairs each
- `client_fn(cid)` — Flower client factory. For each client:
  - Loads global weights from server via `set_parameters()`
  - Builds a local `RAGSystem` with the shared knowledge store
  - Creates a local `HuggingFaceTrainerForLSR`
  - Trains for 1 epoch on its private data split
  - Returns updated weights + `{"loss": actual_training_loss}` via `get_properties()`
- `weighted_average()` — Custom Flower metrics aggregation function that computes per-client and average losses, appends to `ROUND_METRICS` for CSV logging
- `main()` — Runs `fl.simulation.start_simulation()` with 3 clients for 3 rounds, then writes `federated_iid_results.csv`

**Key technical challenge:** Flower's `start_simulation` creates fresh clients each round, but our knowledge store and retriever must be shared. Solved using module-level globals (`KNOWLEDGE_STORE`, `CLIENT_TRAIN_DATA`).

---

### 3. `federated_noniid_lsr.py` (352 lines) — Fed-NonIID Baseline

**Purpose:** FedAvg with heterogeneous data domains. Client 2 has SciFact (science) data instead of NFCorpus (medical).

**Key difference from IID script:**
- `setup_noniid_data()` — Instead of splitting one dataset, loads TWO separate BEIR datasets:
  - Client 0: NFCorpus subset A (first half of medical pairs)
  - Client 1: NFCorpus subset B (second half of medical pairs)
  - Client 2: SciFact (completely different domain — science)
- Builds a shared knowledge store from NFCorpus docs only (the evaluation domain)
- CSV output includes per-client domain labels

**Why this matters:** Shows that cross-domain clients dilute the global model (11% convergence vs 17% for IID), motivating the need for quality-aware and domain-aware aggregation.

---

## QA-FedAvg Scripts (3 scripts)

### 4. `quality_aware_fedavg.py` (125 lines) — ⭐ Core Contribution

**Purpose:** Our novel aggregation strategy. This is the main intellectual contribution of the project.

**Class: `QualityAwareFedAvg(FedAvg)`**

Subclasses Flower's `FedAvg` strategy and overrides `aggregate_fit()`:

1. **Extract client data:** For each client's `FitRes`, extracts model weights (as numpy arrays), number of examples, and training loss from `fit_res.metrics["loss"]`
2. **Compute quality scores:** `q_j = (1/L_j) / Σ(1/L_k)` — normalized inverse loss. Lower loss → higher quality score.
3. **Compute size weights:** `s_j = n_j / N` — standard FedAvg weights.
4. **Combine:** `w_j = α·q_j + (1-α)·s_j`, then normalize to sum to 1.
5. **Aggregate:** Per-layer weighted sum of model parameters.
6. **Store diagnostics:** Saves per-round losses, quality scores, and combined weights for CSV logging.

**Why it's a drop-in replacement:** Only overrides `aggregate_fit()`. Client code doesn't change at all. At α=0, the quality term vanishes and it exactly reproduces FedAvg.

---

### 5. `federated_noisy_qa.py` (399 lines) — Experiment Runner

**Purpose:** Runs the QA-FedAvg experiment with a noisy client to demonstrate α's effectiveness.

**Key function — `split_noisy()`:**
- First splits data IID (same as `split_iid`)
- Then **corrupts Client 2's data**: for 70% of its training pairs, the response is randomly reassigned to a different pair's response. This simulates a client with mislabeled/low-quality data.
- The corruption is deterministic (seeded) for reproducibility.

**How it differs from the IID baseline script:**
- Uses `QualityAwareFedAvg` strategy (imported from `quality_aware_fedavg.py`) instead of Flower's default `FedAvg`
- Accepts `--alpha` command-line argument to configure the quality-awareness parameter
- Accepts `--noise-ratio` to control what fraction of Client 2's data is corrupted
- CSV output includes per-client weights alongside losses

**Experiment flow:**
```
python federated_noisy_qa.py --alpha 0.0   # → baseline FedAvg
python federated_noisy_qa.py --alpha 0.2   # → mild quality awareness
python federated_noisy_qa.py --alpha 0.4   # → moderate
python federated_noisy_qa.py --alpha 0.6   # → optimal (best results)
python federated_noisy_qa.py --alpha 1.0   # → pure quality (over-correction)
```

---

### 6. `run_alpha_sweep.py` (convenience launcher)

A simple script that runs `federated_noisy_qa.py` in a loop across α ∈ {0.0, 0.2, 0.4, 0.6, 1.0} to automate the full alpha sweep experiment.

---

## Output Files

### Baseline CSVs (in `baseline/`)
| File | Columns |
|------|---------|
| `centralized_results.csv` | epoch, loss, mrr, recall_at_k, ndcg_at_k, weight_hash |
| `federated_iid_results.csv` | round, avg_loss, client_0_loss, client_1_loss, client_2_loss, pre_mrr, pre_recall_at_k, pre_ndcg_at_k |
| `federated_noniid_results.csv` | round, avg_loss, client_0_loss/domain, client_1_loss/domain, client_2_loss/domain, pre_mrr, pre_recall_at_k, pre_ndcg_at_k |

### QA-FedAvg CSVs (in `quality_aware_fedrag/output_csv_files/`)
| File | Content |
|------|---------|
| `results_alpha_0.0.csv` | FedAvg baseline (noisy client scenario) |
| `results_alpha_0.2.csv` | α=0.2 — mild quality awareness |
| `results_alpha_0.4.csv` | α=0.4 — moderate quality awareness |
| `results_alpha_0.6.csv` | α=0.6 — optimal (7.3% improvement) |
| `results_alpha_1.0.csv` | α=1.0 — over-correction |

---

## Summary: Lines of Code Contributed

| File | Lines | Category |
|------|------:|----------|
| `prepare_beir_data.py` (baseline) | 180 | Data pipeline |
| `prepare_beir_data.py` (QA) | 276 | Data pipeline (extended) |
| `centralized_lsr.py` | 198 | Baseline experiment |
| `federated_iid_lsr.py` | 318 | Baseline experiment |
| `federated_noniid_lsr.py` | 352 | Baseline experiment |
| `quality_aware_fedavg.py` | 125 | **Core contribution** |
| `federated_noisy_qa.py` | 399 | QA experiment |
| `run_alpha_sweep.py` | ~90 | Automation |
| **Total** | **~1,938** | |

Additionally, ~100 lines of modifications were made to the FedRAG library source code itself (bug fixes to `data_collators/huggingface/lsr.py`, `trainers/huggingface/lsr.py`, and `trainer_managers/huggingface.py`).
