# Experiment 05 — Log

## Config

- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2`
- **Training objective**: `MultipleNegativesRankingLoss` (InfoNCE / contrastive) — replaces LSR + distilgpt2
- **LR**: 2e-6
- **Rounds**: 4, **Epochs**: 1, **Batch size**: 8
- **Clients**: 5, **Split**: equal
- **Noise ladder**: `{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.5, 4: 0.9}`
- **Noise mode**: `random_negative`
- **β**: 5.0
- **Quality signal**: rank-displacement on 400 shared probe pairs
- **Alphas**: 0.0, 0.3, 0.7, 1.0
- **Dataset**: NFCorpus (3633 docs, 4000 train pairs, 400 shared quality, 400 server val, 500 final test)
- **Pre-train baseline**: MRR=0.0243, Recall@10=0.0650, NDCG@10=0.0337

---

## Key Change From Previous Experiments

The LSR training objective (KL-divergence against distilgpt2) was replaced with `MultipleNegativesRankingLoss`. This was required because distilgpt2 assigned uniform log-likelihoods to all domain text, making the training gradient identical across all clients regardless of noise type. With MNR, the training signal is directly determined by the `response` field — a `random_negative` response pushes the query encoder toward an unrelated document, producing a measurably destructive gradient.

**Hash sanity check passed** (1-round pre-run): R1 hash = `b7e63b2c...` ≠ `5706c0d7` (old LSR baseline). Noise is now visible to training.

**Runtime**: ~10 min per alpha run (vs ~60 min with LSR + distilgpt2 inference). Generator removed entirely.

---

## Model Hash Verification

All 4 rounds produced distinct hashes for every alpha — training is no longer stuck.

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| R1 | `b7e63b2c` | `a042ba5b` | `09d38cb8` | `0cf73c97` |
| R2 | `331d0b5e` | `8e529752` | `ee712417` | `02cbf890` |
| R3 | `cef47123` | `e676ad81` | `9eab8c05` | `ae2d257c` |
| R4 | `b4223ef6` | `ffa3e61f` | `0f01e099` | `d81f11d8` |

All 16 hashes are unique across rounds AND across alpha values — each alpha produces a genuinely different aggregated model.

---

## Server Validation MRR (used for best-round selection)

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| Pre-train | 0.02425 | 0.02425 | 0.02425 | 0.02425 |
| R1 | 0.02416 | 0.02416 | 0.02417 | 0.02423 |
| R2 | 0.02427 | 0.02429 | 0.02429 | 0.02429 |
| R3 | **0.02568** | **0.02593** | **0.02590** | **0.02587** |
| R4 | 0.02554 | 0.02545 | 0.02545 | 0.02545 |

Best round = **R3 for all alphas**.

---

## Final Test Metrics (from best round, R3)

| α | Best Round | Best Val MRR | Final Test MRR | Final Test NDCG | vs Pre-train MRR |
|---|-----------|-------------|----------------|-----------------|------------------|
| Pre-train | — | 0.02425 | 0.01974 | 0.03183 | — |
| 0.0 | R3 | 0.02568 | 0.02085 | 0.03267 | **+5.6%** |
| 0.3 | R3 | 0.02593 | **0.02090** | **0.03272** | **+5.9%** |
| 0.7 | R3 | 0.02590 | 0.01983 | 0.03192 | +0.5% |
| 1.0 | R3 | 0.02587 | 0.01993 | 0.03201 | +1.0% |

---

## Per-Client Quality Losses (rank displacement) — Round 3

| Client | Noise | loss (α=0.0) | loss (α=0.3) | loss (α=0.7) | loss (α=1.0) |
|--------|-------|-------------|-------------|-------------|-------------|
| 0 | 0.0% | 0.0075 | 0.0100 | 0.0025 | 0.0013 |
| 1 | 0.0% | 0.0000 | 0.0038 | 0.0000 | -0.0013 |
| 2 | 0.0% | 0.0163 | 0.0138 | 0.0063 | 0.0100 |
| 3 | 50%  | 0.0238 | 0.0175 | 0.0038 | 0.0038 |
| 4 | 90%  | 0.0113 | 0.0163 | 0.0163 | 0.0150 |

---

## Per-Client Weights (combined) — Round 3 (α=0.3)

| Client | Noise | quality_score | size_weight | combined_weight |
|--------|-------|--------------|-------------|-----------------|
| 0 | 0.0% | 0.001838 | 0.20 | 0.14055 |
| 1 | 0.0% | 0.998116 | 0.20 | **0.43944** |
| 2 | 0.0% | 0.000042 | 0.20 | 0.14001 |
| 3 | 50%  | 0.000001 | 0.20 | 0.14000 |
| 4 | 90%  | 0.000003 | 0.20 | 0.14000 |

---

## Acceptance Test Results

| Test | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|------|-------|-------|-------|-------|
| `alpha_zero_matches_fedavg` | N/A | N/A | N/A | N/A |
| `distinct_round_trajectories` | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| `higher_loss_clients_downweighted` | N/A | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| `lower_quality_clients_downweighted` | N/A | ❌ 1/4 | ❌ ?/4 | ❌ 0/4 |

---

## Analysis

### What worked

1. **Training is now genuinely learning** — distinct model hashes, +5.6–5.9% MRR improvement over pre-train for α=0.0/0.3.
2. **`higher_loss_clients_downweighted` passes 4/4 rounds** — the weighting mechanism is mechanically correct.
3. **α=0.3 marginally beats FedAvg** on both final test MRR (0.02090 vs 0.02085) and NDCG (0.03272 vs 0.03267).
4. **Runtime improvement** — 6× faster without distilgpt2 inference.

### What failed

**`lower_quality_clients_downweighted` fails** — this is the primary gate for the contribution.

The rank displacement quality signal is too **low signal-to-noise** to consistently separate clean from noisy clients:
- Quality losses across all 5 clients span only ~0.01–0.03, with overlapping ranges
- Clean client 2 often has higher loss than noisy client 4 — the ordering is not reliable
- With β=5.0, even small differences cause extreme weight concentration on the winner

### The client 1 dominance problem

Client 1 (clean) consistently produces the most negative rank displacement (improves rankings most). With β=5.0 and α=1.0, it receives **99.98% of aggregation weight in Round 1**, making the global model essentially a single-client update. This explains why α=0.7 and α=1.0 perform **worse than FedAvg** on final test — high-α runs collapse to client 1's perspective.

The fundamental issue: **β=5.0 is far too aggressive for this signal magnitude.** A quality loss difference of 0.003 between client 1 and the next-best client produces a 1000:1 weight ratio. Even if the ordering were correct, this concentration is too extreme.

### Root cause summary

Two compounding problems:
1. **Rank displacement signal has too much variance** — differences of 0.001–0.03 are within round-to-round noise for a 400-pair probe set. The signal occasionally gets the noisy/clean ordering right but not consistently.
2. **β=5.0 is too aggressive** — it amplifies any small quality advantage into near-total weight dominance, making the system unstable instead of robustly downweighting noisy clients.

---

## Next Steps (Step 6)

Two targeted fixes, to be implemented together:

1. **Switch quality signal to delta MRR**: replace rank displacement with `loss_i = MRR_before − MRR_after` computed on the shared probe set. This is directly aligned with the evaluation metric, has no unjustified asymmetry coefficient (the current 0.5 factor is arbitrary), and is easier to interpret and defend.

2. **Reduce β from 5.0 to 1.0 or 2.0**: prevent single-client weight dominance. With β=1.0, a client with quality_loss=0.003 vs 0.010 gets ~2× weight instead of 1000×. This makes the aggregation robust and the α curve interpretable.

Store results in `exp_06_results/`.
