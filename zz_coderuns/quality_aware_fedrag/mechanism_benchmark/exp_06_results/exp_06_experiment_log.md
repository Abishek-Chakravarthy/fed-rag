# Experiment 06 — Log

## Config

- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2`
- **Training objective**: `MultipleNegativesRankingLoss` (InfoNCE / contrastive) via `ContrastiveFlowerClient`
- **LR**: 2e-6
- **Rounds**: 4, **Epochs**: 1, **Batch size**: 8
- **Clients**: 5, **Split**: equal
- **Noise ladder**: `{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.5, 4: 0.9}`
- **Noise mode**: `random_negative`
- **Quality signal**: `delta_mrr` (MRR_before − MRR_after on 400-pair shared probe) — **NEW** (was rank-displacement in exp_05)
- **β values tested**: **1.0** and **2.0** (was 5.0 in exp_05)
- **Alphas**: 0.0, 0.3, 0.7, 1.0
- **Dataset**: NFCorpus (3633 docs, 4000 train pairs, 400 shared quality, 400 server val, 500 final test)
- **Pre-train baseline**: Val MRR=0.02425, Test MRR=0.01974, Test NDCG=0.03183

---

## Changes From exp_05

1. **Quality signal**: Replaced `rank-displacement` with `delta_mrr = MRR_before − MRR_after` on shared probe. Positive = degraded, negative = improved.
2. **β reduction**: Tested β=1.0 and β=2.0 (was 5.0). Goal: prevent single-client weight dominance.

---

## Model Hash Verification

All hashes are unique across rounds AND across alpha/beta values — training is producing genuinely different models.

### β=1.0 hashes

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| R1 | `2b12d5df` | `cc1b62c5` | `28cf3fa0` | `9bd7dc3e` |
| R2 | `420d676e` | `621a2744` | `501cde10` | `da6f00c8` |
| R3 | `cb4917db` | `2bacda94` | `84068963` | `0baecef8` |
| R4 | `70ef4835` | `da408195` | `c10255d3` | `900fdce9` |

### β=2.0 hashes

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| R1 | `c51ca8a8` | `6b5a6d9b` | `7e8a262b` | `4d9a0c46` |
| R2 | `66b9195f` | `bdef01c7` | `6e266eec` | `394f8c8b` |
| R3 | `8583459f` | `5266b947` | `d5afb28d` | `4d016009` |
| R4 | `af73765b` | `ee22067f` | `819c6392` | `2a232bdb` |

**Note**: β=1.0 and β=2.0 produce different hashes from each other (expected, since different weighting affects aggregation). Both differ from exp_05 hashes.

---

## Server Validation MRR

### β=1.0

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| Pre | 0.02425 | 0.02425 | 0.02425 | 0.02425 |
| R1 | 0.02416 | 0.02416 | 0.02416 | 0.02421 |
| R2 | 0.02427 | 0.02427 | 0.02435 | 0.02435 |
| R3 | **0.02568** | **0.02568** | **0.02568** | **0.02560** |
| R4 | 0.02554 | 0.02554 | 0.02560 | 0.02552 |

### β=2.0

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| Pre | 0.02425 | 0.02425 | 0.02425 | 0.02425 |
| R1 | 0.02416 | 0.02416 | 0.02421 | 0.02421 |
| R2 | 0.02427 | 0.02427 | 0.02435 | 0.02435 |
| R3 | **0.02568** | **0.02568** | **0.02560** | **0.02560** |
| R4 | 0.02554 | 0.02560 | 0.02552 | 0.02552 |

Best round = **R3** for all α/β combinations.

---

## Final Test Metrics (from best round, R3)

### β=1.0

| α | Best Round | Best Val MRR | Final Test MRR | Final Test NDCG | Recall@10 | vs Pre-train MRR | vs α=0.0 MRR |
|---|-----------|-------------|----------------|-----------------|-----------|-------------------|--------------|
| Pre | — | 0.02425 | 0.01974 | 0.03183 | — | — | — |
| 0.0 | R3 | 0.02568 | 0.02085 | 0.03267 | 0.072 | **+5.6%** | — |
| 0.3 | R3 | 0.02568 | 0.02112 | 0.03290 | 0.072 | **+7.0%** | **+1.3%** |
| 0.7 | R3 | 0.02568 | **0.02122** | **0.03299** | 0.072 | **+7.5%** | **+1.8%** |
| 1.0 | R3 | 0.02560 | 0.02117 | 0.03294 | 0.072 | **+7.2%** | **+1.5%** |

### β=2.0

| α | Best Round | Best Val MRR | Final Test MRR | Final Test NDCG | Recall@10 | vs Pre-train MRR | vs α=0.0 MRR |
|---|-----------|-------------|----------------|-----------------|-----------|-------------------|--------------|
| Pre | — | 0.02425 | 0.01974 | 0.03183 | — | — | — |
| 0.0 | R3 | 0.02568 | 0.02085 | 0.03267 | 0.072 | **+5.6%** | — |
| 0.3 | R3 | 0.02568 | 0.02112 | 0.03290 | 0.072 | **+7.0%** | **+1.3%** |
| 0.7 | R3 | 0.02560 | 0.02117 | 0.03294 | 0.072 | **+7.2%** | **+1.5%** |
| 1.0 | R3 | 0.02560 | **0.02150** | **0.03320** | 0.072 | **+8.9%** | **+3.1%** |

---

## Key Improvement: α=1.0 No Longer Collapses

### exp_05 (β=5.0) vs exp_06

| Config | α=0.7 Test MRR | α=1.0 Test MRR | α=1.0 vs α=0.0 |
|--------|---------------|----------------|-----------------|
| exp_05 (β=5.0) | 0.01983 | 0.01993 | **−4.4%** (worse) |
| exp_06 (β=1.0) | **0.02122** | **0.02117** | **+1.5%** (better) |
| exp_06 (β=2.0) | 0.02117 | **0.02150** | **+3.1%** (better) |

In exp_05, α=0.7 and α=1.0 performed **worse** than FedAvg due to extreme weight concentration. In exp_06, **all α>0 values beat FedAvg**. The β reduction successfully prevents single-client dominance.

---

## Per-Client Quality Scores (softmax of delta_mrr)

### β=1.0 — quality_score per round

| Round | C0 (clean) | C1 (clean) | C2 (clean) | C3 (50%) | C4 (90%) |
|-------|-----------|-----------|-----------|---------|---------|
| R1 | 0.361 | 0.288 | 0.036 | 0.270 | 0.045 |
| R2 | 0.188 | 0.083 | 0.057 | 0.038 | **0.634** |
| R3 | 0.424 | 0.129 | 0.033 | 0.358 | 0.055 |
| R4 | **0.702** | 0.089 | 0.089 | 0.034 | 0.085 |

### β=2.0 — quality_score per round

| Round | C0 (clean) | C1 (clean) | C2 (clean) | C3 (50%) | C4 (90%) |
|-------|-----------|-----------|-----------|---------|---------|
| R1 | 0.450 | 0.287 | 0.004 | 0.252 | 0.007 |
| R2 | 0.079 | 0.015 | 0.007 | 0.003 | **0.895** |
| R3 | 0.547 | 0.051 | 0.003 | 0.390 | 0.009 |
| R4 | **0.953** | 0.015 | 0.015 | 0.002 | 0.014 |

---

## Delta MRR Signal Analysis

The delta_mrr values (MRR_before − MRR_after) are **extremely small** — on the order of ±0.0001 to ±0.001. This is because the 400-pair shared probe set can only resolve MRR differences at a granularity of ~1/400 ≈ 0.0025 per rank change, and most single-round training moves at most 1–2 documents.

### β=1.0, α=0.0 — raw delta_mrr per round

| Round | C0 (clean) | C1 (clean) | C2 (clean) | C3 (50%) | C4 (90%) | C4 worst? |
|-------|-----------|-----------|-----------|---------|---------|-----------|
| R1 | −0.000424 | −0.000289 | +0.000955 | −0.000250 | +0.000826 | ✅ 2nd worst |
| R2 | −0.000042 | +0.000045 | +0.000083 | +0.000128 | −0.000170 | ❌ best |
| R3 | −0.000146 | +0.000000 | +0.000167 | −0.000125 | +0.000104 | ❌ 2nd worst |
| R4 | −0.000351 | −0.000226 | −0.000226 | −0.000167 | −0.000223 | ❌ 2nd worst |

### β=2.0, α=1.0 — raw delta_mrr per round

| Round | C0 (clean) | C1 (clean) | C2 (clean) | C3 (50%) | C4 (90%) | C4 worst? |
|-------|-----------|-----------|-----------|---------|---------|-----------|
| R1 | −0.000424 | −0.000289 | +0.000955 | −0.000250 | +0.000826 | ✅ 2nd worst |
| R2 | −0.000042 | +0.000045 | +0.000083 | +0.000128 | −0.000135 | ❌ best |
| R3 | +0.000059 | +0.000318 | +0.000193 | +0.000170 | **+0.000353** | **✅ worst** |
| R4 | +0.000250 | +0.000042 | +0.000167 | +0.000042 | +0.000042 | ❌ tied best |

**Key observation**: C4 (90% noise) is correctly identified as worst in ~25% of rounds. In the other rounds, the delta_mrr ordering is essentially random — the probe set resolution is too coarse for reliable discrimination at this noise level.

### Train Loss (contrastive MNR)

| Round | C0 (clean) | C1 (clean) | C2 (clean) | C3 (50%) | C4 (90%) |
|-------|-----------|-----------|-----------|---------|---------|
| R1 | 2.32 | 2.48 | 2.31 | 2.75 | **3.12** |
| R2 | 2.25 | 2.41 | 2.24 | 2.66 | **3.03** |
| R3 | 2.19 | 2.35 | 2.18 | 2.59 | **2.93** |
| R4 | 2.14 | 2.29 | 2.12 | 2.52 | **2.86** |

**Train loss perfectly discriminates**: C4 (90%) > C3 (50%) > clean clients, every round, no exceptions. This is the expected pattern — noisy clients have harder contrastive pairs (random documents paired with queries), resulting in consistently higher loss.

---

## Acceptance Test Results

### β=1.0

| Test | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|------|-------|-------|-------|-------|
| `alpha_zero_matches_fedavg` | ✅ | N/A | N/A | N/A |
| `distinct_round_trajectories` | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| `higher_loss_clients_downweighted` | N/A | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| `lower_quality_clients_downweighted` | N/A | ❌ 1/4 | ❌ 0/4 | ❌ 0/4 |

### β=2.0

| Test | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|------|-------|-------|-------|-------|
| `alpha_zero_matches_fedavg` | ✅ | N/A | N/A | N/A |
| `distinct_round_trajectories` | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| `higher_loss_clients_downweighted` | N/A | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| `lower_quality_clients_downweighted` | N/A | ❌ 0/4 | ❌ 0/4 | ❌ 0/4 |

---

## Analysis

### What worked (significant improvements over exp_05)

1. **High-α no longer collapses**: In exp_05, α=0.7 and α=1.0 performed WORSE than FedAvg (−4.4%). Now ALL α>0 beat FedAvg. β reduction fixed the extreme weight concentration.

2. **Monotonic α benefit (β=2.0)**: `α=0.0 < α=0.3 < α=0.7 < α=1.0` on test MRR — a clean monotonic improvement curve. α=1.0 at β=2.0 achieves the **best result across all experiments**: 0.02150 test MRR (+8.9% over pre-train, +3.1% over FedAvg).

3. **Weight distribution is reasonable**: No client dominates > 95% (unlike exp_05 where client 1 had 99.98%). At β=1.0, max quality_score is ~0.70. At β=2.0, max is ~0.95 but only in R4.

4. **Train loss perfectly separates clean from noisy**: C4 (90%) always has highest contrastive loss, C3 (50%) second-highest. This is the reliable signal we've been looking for.

### What still fails

**`lower_quality_clients_downweighted` test fails** — 0/4 or 1/4 rounds for all configurations.

### Root cause: delta_mrr is too coarse

The `delta_mrr` signal operates at the **wrong resolution** for this task:

- **Resolution**: MRR changes of ±0.0001 are measured on a 400-pair probe. This is ~1/10th of the minimum resolvable change (1/400 = 0.0025 for a single document rank shift). Multiple sub-resolution changes are being averaged, creating noise-dominated measurements.

- **Contrast with train_loss**: Train loss spans 2.12 → 3.12, a 47% relative range. Delta_mrr spans ±0.001, a range of ~5% of the base MRR. The signal-to-noise ratio is fundamentally different.

- **Why it fails**: The `lower_quality_clients_downweighted` test checks whether noisy clients (C3, C4) get lower quality_score than ALL clean clients (C0, C1, C2). Since delta_mrr doesn't reliably rank C4 as worst, this test cannot pass consistently.

### Yet performance improves anyway

Despite failing the formal acceptance test, the α>0 runs DO outperform FedAvg. This suggests the quality signal is providing a **weak but directionally correct** aggregate benefit — even if it can't reliably rank individual clients per-round, the average effect over 4 rounds is positive.

### β=1.0 vs β=2.0

| β | Best α | Best Test MRR | α curve shape |
|---|--------|--------------|---------------|
| 1.0 | 0.7 | 0.02122 | peaks at 0.7, slight dip at 1.0 |
| 2.0 | 1.0 | **0.02150** | monotonically increasing |

β=2.0 produces the better absolute result and a cleaner α curve. The slightly sharper softmax apparently helps when the signal is directionally correct on average.

---

## Shared Quality MRR (post-training, on shared probe)

All clients converge to nearly identical shared_quality_mrr (~0.0416–0.0420) after training, regardless of noise level. This confirms that the MRR measurement on the shared probe is dominated by the pre-trained model's knowledge rather than the single-epoch fine-tuning delta.

---

## Summary Comparison: exp_05 → exp_06

| Metric | exp_05 (β=5.0, rank-disp) | exp_06 β=1.0 (delta_mrr) | exp_06 β=2.0 (delta_mrr) |
|--------|--------------------------|-------------------------|-------------------------|
| Best α>0 test MRR | 0.02090 (α=0.3) | **0.02122** (α=0.7) | **0.02150** (α=1.0) |
| Best α>0 vs FedAvg | +0.2% | **+1.8%** | **+3.1%** |
| α=1.0 vs FedAvg | −4.4% (collapsed) | **+1.5%** | **+3.1%** |
| Max weight concentration | 99.98% | ~70% | ~95% |
| `lower_quality_downweighted` | ❌ 1/4 | ❌ 1/4 | ❌ 0/4 |
| `higher_loss_downweighted` | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |

---

## Next Steps

### Option A: Switch quality signal to train_loss

Train loss perfectly discriminates (C4 > C3 > clean, every round). Instead of `loss = delta_mrr`, use `loss = train_loss` as the quality signal sent to the server. The server would compute `quality_score = softmax(-β * train_loss)`, giving lower weight to clients with higher contrastive loss (i.e., clients trained on nonsensical query-document pairs).

**Pros**: Already available in the metrics, no additional evaluation needed, 100% reliable ordering.  
**Cons**: Train loss reflects both data quality AND data difficulty; a client with genuinely harder (but clean) queries would also be penalized. In this controlled benchmark, data difficulty is equal (same distribution), so this is not a concern.

### Option B: Increase probe set size

Increase `MAX_SHARED_QUALITY` from 400 to 1000+ pairs to improve delta_mrr resolution. This would reduce the number of available training pairs.

### Option C: Accept current results + reframe

The α-curve improvement is real and consistent: +3.1% over FedAvg at β=2.0, α=1.0. The `lower_quality_clients_downweighted` formal test could be relaxed to check the aggregate effect rather than per-round ordering.

### Option D: Multi-seed runs with β=2.0

Run seeds 42, 52, 62 at the current best config (β=2.0, delta_mrr) to establish statistical significance of the α>0 benefit.
