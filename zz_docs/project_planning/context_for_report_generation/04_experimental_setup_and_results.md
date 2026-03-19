# 04 — Experimental Results (Complete Data)

## 1. Baseline Experiments

### 1.1 Centralized Training (Upper Bound)
All 250 training pairs pooled, trained for 3 epochs. Single model, no federation.

| Epoch | Training Loss | MRR | Recall@10 | NDCG@10 | Weight Hash |
|-------|---------------|-----|-----------|---------|-------------|
| 0 (pre-training) | — | 0.064305 | 0.120482 | 0.077805 | 9db17c8aa125 |
| 1 | 0.001037 | 0.064305 | 0.120482 | 0.077805 | 2b7181d66bee |
| 2 | 0.000873 | 0.065309 | 0.120482 | 0.078640 | f1fdc2f5057f |
| 3 | 0.000739 | 0.066432 | 0.132530 | 0.082052 | 0ebf30944e7f |

**Key observations:**
- Loss converges from 0.001037 → 0.000739 (**29% reduction**)
- MRR improves from 0.0643 → 0.0664 (+3.3%)
- Recall@10 improves from 0.1205 → 0.1325 (+10%)
- Weight hashes change each epoch, confirming the model actually updates

### 1.2 Federated IID (Standard FedAvg)
500 training pairs split equally across 3 clients (IID — random shuffle). 3 federated rounds.

| Round | Avg Loss | Client 0 Loss | Client 1 Loss | Client 2 Loss | Pre-MRR | Pre-Recall@10 | Pre-NDCG@10 |
|-------|----------|---------------|---------------|---------------|---------|---------------|-------------|
| 1 | 0.001066 | 0.001037 | 0.001072 | 0.001089 | 0.064305 | 0.120482 | 0.077805 |
| 2 | 0.000971 | 0.000948 | 0.000992 | 0.000975 | 0.064305 | 0.120482 | 0.077805 |
| 3 | 0.000882 | 0.000883 | 0.000898 | 0.000866 | 0.064305 | 0.120482 | 0.077805 |

**Key observations:**
- Loss converges from 0.001066 → 0.000882 (**17% reduction**)
- All client losses are similar (IID data → similar distributions)
- Retrieval metrics remain flat at pre-training levels (3 rounds insufficient to significantly change retrieval behavior)

### 1.3 Federated Non-IID (Standard FedAvg, Heterogeneous Domains)
3 clients with different data domains:
- Client 0: NFCorpus subset A (Medical)
- Client 1: NFCorpus subset B (Medical, different queries)
- Client 2: SciFact (Science — completely different domain)

| Round | Avg Loss | Client 0 (Medical-A) | Client 1 (Medical-B) | Client 2 (SciFact) | Pre-MRR | Pre-Recall@10 | Pre-NDCG@10 |
|-------|----------|---------------------|---------------------|-------------------|---------|---------------|-------------|
| 1 | 0.001008 | 0.000850 | 0.001138 | 0.001037 | 0.064305 | 0.120482 | 0.077805 |
| 2 | 0.000951 | 0.000973 | 0.001059 | 0.000820 | 0.064305 | 0.120482 | 0.077805 |
| 3 | 0.000893 | 0.000976 | 0.000913 | 0.000791 | 0.064305 | 0.120482 | 0.077805 |

**Key observations:**
- Loss converges from 0.001008 → 0.000893 (**11% reduction** — slowest of all methods)
- SciFact client (Client 2) has consistently different loss patterns from Medical clients
- This demonstrates the "FL Gap" — cross-domain data dilutes the global model

### 1.4 Baseline Comparison Summary

| Method | R1 Loss | R3 Loss | ΔLoss | Loss Reduction |
|--------|---------|---------|-------|---------------|
| Centralized | 0.001037 | 0.000739 | -0.000298 | **29%** |
| Fed-IID | 0.001066 | 0.000882 | -0.000184 | **17%** |
| Fed-NonIID | 0.001008 | 0.000893 | -0.000115 | **11%** |

**The "FL Gap"**: Centralized > Fed-IID > Fed-NonIID in convergence speed. The non-IID gap is 18 percentage points worse than centralized — this is the problem QA-FedAvg targets.

---

## 2. QA-FedAvg Experiments (Alpha Sweep)

### Scenario
3 clients, 3 rounds. Client 0 and Client 1 have clean, high-quality data. Client 2 has **noisy data** (corrupted query-document pairs) resulting in significantly higher training loss.

### 2.1 Global Loss Convergence

| α | Round 1 Loss | Round 2 Loss | Round 3 Loss | Δ vs FedAvg (R3) |
|---|-------------|-------------|-------------|-------------------|
| **0.0 (FedAvg)** | 0.001183 | 0.001068 | **0.000973** | — (baseline) |
| 0.2 | 0.001167 | 0.001044 | 0.000943 | -3.1% |
| 0.4 | 0.001153 | 0.001022 | 0.000918 | -5.7% |
| **0.6 (Optimal)** | **0.001144** | **0.001005** | **0.000902** | **-7.3%** |
| 1.0 | 0.001157 | 0.001033 | 0.000932 | -4.2% |

**Inverted-U pattern**: Performance improves with α up to 0.6, then degrades at α=1.0 due to over-correction. α=0.6 achieves the optimal balance.

### 2.2 Per-Client Training Losses (Round 3)

| α | Client 0 (Clean) | Client 1 (Clean) | Client 2 (Noisy) |
|---|-----------------|-----------------|-----------------|
| 0.0 | 0.000782 | 0.000808 | 0.001325 |
| 0.2 | 0.000752 | 0.000778 | 0.001295 |
| 0.4 | 0.000728 | 0.000755 | 0.001268 |
| 0.6 | 0.000715 | 0.000742 | 0.001245 |
| 1.0 | 0.000742 | 0.000768 | 0.001282 |

**Observation**: Client 2 (noisy) consistently has ~1.7x higher loss than clean clients. With higher α, clean client losses decrease faster because the global model is less polluted by noisy updates.

### 2.3 Client Weight Redistribution (Round 3)

| α | Client 0 Weight | Client 1 Weight | Client 2 Weight | Client 2 Influence Change |
|---|:-:|:-:|:-:|---|
| **0.0 (FedAvg)** | 0.332 | 0.332 | **0.336** | Baseline (full influence) |
| 0.2 | 0.355 | 0.350 | **0.295** | -12% |
| 0.4 | 0.373 | 0.366 | **0.261** | -22% |
| **0.6 (Optimal)** | 0.390 | 0.382 | **0.228** | **-32%** |
| 1.0 | 0.415 | 0.405 | **0.180** | -46% |

**Key finding**: QA-FedAvg automatically detects the noisy client through its higher training loss and progressively reduces its aggregation influence. At α=0.6, the noisy client's weight drops by 32% — enough to significantly improve the global model without over-correcting.

### 2.4 Interpretation of Over-Correction at α=1.0
At α=1.0, Client 2's weight drops to 0.180 (46% reduction), but global loss (0.000932) is actually worse than α=0.4 (0.000918). This is because:
- Quality scores derived from a single epoch of local training contain noise
- Completely ignoring dataset size removes a stabilizing factor
- The optimal strategy balances quality awareness with size-based regularization
