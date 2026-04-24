# Improvement Plan — QA-FedAvg Mechanism Benchmark

> Anchored to `latest_results/experiment_log.md`. Update the log after every run.

---

## What Run 1 Told Us

**Config**: `paraphrase-MiniLM-L3-v2`, lr=2e-6, 4 rounds, 1 epoch, 5 clients, 4000 train, 8000 docs, equal split, shuffle, noise ladder 0/0/0/0.3/0.7, β=5.0, rank-displacement quality signal.

**Two problems found**:
1. **Training is destructive** — pre-train MRR (0.0237) > best final MRR (0.0234 at α=1.0). Best-round selection picks round 1 or 2 because the retriever only gets worse.
2. **Quality signal doesn't discriminate** — rank-displacement differences between clean and noisy clients are tiny (~0.01). Acceptance test `lower_quality_clients_downweighted` fails for all α>0.

**One positive**: the weighting mechanism itself works correctly (α=0 gives equal weights, α=1 gives ~99% to the "best" client).

---

## Step 1: Non-Federated Diagnostic ✅ COMPLETED

**Question**: Can LSR training improve retrieval on NFCorpus at all, for either retriever?

**Results** (from `lsr_training_check/diagnostic_summary.json`):

| ID | Retriever | LR | Pre MRR | Post MRR | Δ% | Verdict |
|----|-----------|-----|---------|----------|-----|---------|
| 1a | `paraphrase-MiniLM-L3-v2` | 2e-6 | 0.0160 | 0.0149 | **-7.1%** | ❌ DEGRADED |
| 1b | `paraphrase-MiniLM-L3-v2` | 5e-7 | 0.0160 | 0.0158 | **-1.5%** | ❌ DEGRADED |
| 1c | `all-MiniLM-L6-v2` | 2e-6 | 0.0243 | 0.0268 | **+10.3%** | ✅ IMPROVED |
| 1d | `all-MiniLM-L6-v2` | 5e-7 | 0.0243 | 0.0242 | **-0.04%** | ❌ DEGRADED |

**Gate**: ✅ **PASSED** — Variant 1c shows +10.3% relative MRR improvement.

**Winning config**: `all-MiniLM-L6-v2`, lr=2e-6.

---

## Step 2: Federated α-Sweep With Winning Config ✅ COMPLETED

**Config change**: Switched `MECHANISM_RETRIEVER_MODEL` to `all-MiniLM-L6-v2`.

**Results** (from `latest_results/`, Run 2 in experiment_log):

### Server validation MRR (used for best-round selection)

| Round | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|-------|-------|-------|-------|-------|
| Pre-train | 0.0243 | 0.0243 | 0.0243 | 0.0243 |
| R1 | 0.0249 | 0.0248 | 0.0246 | 0.0247 |
| R2 | **0.0262** | **0.0262** | 0.0265 | 0.0265 |
| R3 | 0.0257 | 0.0257 | **0.0275** | **0.0272** |
| R4 | 0.0235 | 0.0233 | 0.0257 | 0.0261 |

### Final test MRR

| α | Final Test MRR | Final Test NDCG |
|---|----------------|-----------------|
| 0.0 | 0.0202 | 0.0322 |
| 0.3 | 0.0200 | 0.0320 |
| 0.7 | 0.0200 | 0.0328 |
| 1.0 | 0.0201 | 0.0333 |

### Gate assessment

**Gate 1** (α=0.0 final MRR above pre-train): ✅ **PASSED**
- Pre-train test MRR: 0.0197 → α=0.0 final test MRR: 0.0202 (+2.3%)
- Federated LSR training no longer degrades the retriever — the retriever switch fixed the fundamental issue.

**Gate 2** (α>0 beats α=0.0): ⚠️ **PARTIALLY PASSED**
- On **server validation MRR**: α=0.7 (0.0275, best at R3) and α=1.0 (0.0272, best at R3) clearly beat α=0.0 (0.0262, best at R2). Higher α → model stays useful for one more round before degrading. This is a real signal.
- On **final test MRR**: all alphas produce ~0.020 — flat. The test set is too small (400 pairs) or too different from the validation set to reflect the val-side advantage.

### Key observations

1. **α>0 delays degradation by ~1 round**: α=0.0 and α=0.3 peak at R2 and degrade by R3. α=0.7 and α=1.0 peak at R3 and degrade by R4. The quality weighting is giving noisy clients less influence, which preserves the model for longer.
2. **The quality signal still doesn't discriminate clean vs noisy correctly**: `lower_quality_clients_downweighted` fails (2/4 or 1/4 rounds pass). The rank-displacement values for all 5 clients are nearly identical — clean and noisy clients both show similar rank shifts.
3. **α=1.0 NDCG (0.0333) is the highest across all alphas**, beating α=0.0 NDCG (0.0322) by 3.4%. This is small but consistent with the mechanism working at a micro level.

---

## Step 3: Increase Noise Separation ← **MUST RE-RUN**

**Why**: The quality signal fails to discriminate because 30% and 70% shuffle noise don't create enough retrieval quality difference between clients. The rank-displacement values are nearly identical across all 5 clients.

**What to change**: Noise ladder from `{0, 0, 0, 0.3, 0.7}` to `{0, 0, 0, 0.5, 0.9}`.

**What to update in `federated_noisy_qa.py`**: ✅ Already done (commit `23a9643`).

**exp_03 attempt**: ⚠️ **INVALID** — model hashes are identical to exp_02 across all 16 runs (4α × 4R). The Kaggle notebooks cloned an older version of `q-fedrag2` before the noise change was pushed. The manifests log the intended 0.5/0.9 values, but the actual training used 0.3/0.7.

**Action**: Verify the latest commit is on the remote, then re-run all 4 alpha notebooks on Kaggle. Store results in `exp_03b_results/`.

**Run**: Full α sweep (0.0, 0.3, 0.7, 1.0)

**Gate**:
1. `lower_quality_clients_downweighted` passes for at least one α>0 in at least 3 out of 4 rounds
2. At least one α>0 beats α=0.0 on final test MRR or NDCG
3. **Sanity check**: α=0.0 model hashes must differ from exp_02 (confirming different training data)

- **Pass** → run multi-seed. 🎉
- **Fail** → proceed to Step 4.

---

## Step 4: Change Quality Signal

**Why**: If rank-displacement can't discriminate even with 90% noise, the signal itself may be fundamentally too noisy for this setup.

**What to change**: Replace rank-displacement with **holdout evaluation MRR** as the quality signal sent to the server.
- Each client already has a clean holdout set (160 pairs)
- After local training, evaluate the client's retriever on its holdout
- Send `loss = -holdout_mrr` to the server (lower MRR → higher loss → lower weight under QA-FedAvg)
- This is a much more direct signal — noisy training should clearly hurt holdout retrieval

**Run**: Full α-sweep with the stronger noise from Step 3.

**Gate**: Same as Step 3.

- **Pass** → run multi-seed. 🎉
- **Fail** → the noise levels may be too subtle even for direct evaluation. Reframe the contribution as demonstrating the mechanism under extreme noise, or pivot to DAS-FedAvg as the primary contribution.

---

## Decision Tree

```
Step 1: Can LSR training improve retrieval at all?  ✅ YES (variant 1c: +10.3%)
  │
  └── Step 2: Does federated LSR still improve? Does α help?  ✅ PARTIALLY
              │  Training works (Gate 1 ✅). α>0 delays degradation by 1 round (Gate 2 ✅ on val).
              │  But final test MRR is flat. Quality signal doesn't discriminate noisy clients.
              │
              └── Step 3: Increase noise to 0/0/0/0.5/0.9.  ← YOU ARE HERE
                           │
                           ├── Quality signal discriminates → Done! 🎉
                           │
                           └── Still flat → Step 4: Change quality signal.
                                             │
                                             ├── Works → Done! 🎉
                                             └── Fails → Reframe contribution.
```

---

## Do NOT Repeat

- ❌ `paraphrase-MiniLM-L3-v2` for mechanism benchmark (diagnostic proved it can't learn from LSR)
- ❌ LR=5e-7 (diagnostic showed negligible movement for both retrievers)
- ❌ LR=5e-6 (known to cause collapse)
- ❌ Epochs ≥ 2 (known to cause collapse)
- ❌ Rounds ≥ 6 (known to cause degradation)
- ❌ Hard-negative noise before shuffle works (known to be always flat)
- ❌ Mixed noise mode (known to fail)
- ❌ Noise ladder 0/0/0/0.3/0.7 with `all-MiniLM-L6-v2` (Run 2: quality signal doesn't discriminate)
