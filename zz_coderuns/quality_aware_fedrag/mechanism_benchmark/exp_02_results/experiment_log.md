# QA-FedAvg Mechanism Benchmark — Experiment Log

> **Purpose**: Prevent going in circles. Every run batch gets one row. Check here BEFORE trying a new config.

---

## Runs (newest first)

| # | Date | Retriever | LR | Rounds | Epochs | Clients | Train | Docs | Split | Noise Levels | β | Quality Signal | Best α MRR | FedAvg MRR | Pre-train MRR | Verdict |
|---|------|-----------|-----|--------|--------|---------|-------|------|-------|-------------|---|----------------|------------|------------|---------------|---------|
| **2** | Apr 24, 2026 | `all-MiniLM-L6-v2` | 2e-6 | 4 | 1 | 5 | 4000 | 8000 | equal | 0,0,0,0.3,0.7 | 5.0 | rank-displacement | 0.0272 (α=1.0, R3) | 0.0262 (α=0.0, R2) | **0.0243** | ⚠️ Training improves over pre-train (Gate 1 ✅). α=0.7 and α=1.0 show val_MRR above α=0.0 (Gate 2 ✅ on val). But final_test_MRR is flat across all α (~0.020). Quality signal still doesn't discriminate noisy clients (`lower_quality_clients_downweighted` fails for all α>0). |
| **1** | Apr 24, 2026 | `paraphrase-MiniLM-L3-v2` | 2e-6 | 4 | 1 | 5 | 4000 | 8000 | equal | 0,0,0,0.3,0.7 | 5.0 | rank-displacement | 0.0234 (α=1.0) | 0.0226 (α=0.0) | **0.0237** | ❌ Training degrades retriever (final < pre-train). Quality signal doesn't discriminate. |

---

## Diagnostic Results (Step 1)

| ID | Retriever | LR | Pre MRR | Post MRR | Δ% | Verdict |
|----|-----------|-----|---------|----------|-----|---------|
| 1a | `paraphrase-MiniLM-L3-v2` | 2e-6 | 0.0160 | 0.0149 | -7.1% | ❌ DEGRADED |
| 1b | `paraphrase-MiniLM-L3-v2` | 5e-7 | 0.0160 | 0.0158 | -1.5% | ❌ DEGRADED |
| 1c | `all-MiniLM-L6-v2` | 2e-6 | 0.0243 | 0.0268 | **+10.3%** | ✅ IMPROVED |
| 1d | `all-MiniLM-L6-v2` | 5e-7 | 0.0243 | 0.0242 | -0.04% | ❌ DEGRADED |

---

## Run 2 Detailed Metrics

### Per-round server validation MRR (best-round selection basis)

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| Pre-train | 0.0243 | 0.0243 | 0.0243 | 0.0243 |
| R1 | 0.0249 | 0.0248 | 0.0246 | 0.0247 |
| R2 | **0.0262** | **0.0262** | 0.0265 | 0.0265 |
| R3 | 0.0257 | 0.0257 | **0.0275** | **0.0272** |
| R4 | 0.0235 | 0.0233 | 0.0257 | 0.0261 |
| Best round | R2 | R2 | R3 | R3 |

### Final test MRR (evaluated at selected best round)

| α | Final Test MRR | Final Test NDCG | Δ vs pre-train |
|---|----------------|-----------------|----------------|
| 0.0 | 0.0202 | 0.0322 | +2.3% |
| 0.3 | 0.0200 | 0.0320 | +1.3% |
| 0.7 | 0.0200 | 0.0328 | +1.5% |
| 1.0 | 0.0201 | 0.0333 | +1.6% |

### Acceptance tests

| Test | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|------|-------|-------|-------|-------|
| alpha_zero_matches_fedavg | ✅ | n/a | n/a | n/a |
| distinct_round_trajectories | ✅ | ✅ | ✅ | ✅ |
| higher_loss_clients_downweighted | n/a | ✅ | ✅ | ✅ |
| lower_quality_clients_downweighted | n/a | ❌ (2/4) | ❌ (2/4) | ❌ (1/4) |

---

## What Has NOT Been Tried (within the mechanism benchmark redesign)

- Stronger noise levels (e.g., 0/0/0/0.5/0.9)
- Fewer training pairs (e.g., 2000 like the diagnostic)
- Using holdout evaluation loss as the quality signal instead of rank-displacement
- More rounds (e.g., 6) to give α>0 more time to diverge — BUT note: pre-redesign history showed ≥6 rounds cause degradation
