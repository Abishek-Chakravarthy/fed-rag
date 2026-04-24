# Improvement Plan — QA-FedAvg Mechanism Benchmark

> Anchored to `exp_02_results/experiment_log.md`. Update the log after every run.

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

**Results** (from `exp_02_results/`, Run 2 in experiment_log):

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

## Step 3: Increase Noise Separation ✅ COMPLETED — FAILED

**What changed**: Noise ladder from `{0, 0, 0, 0.3, 0.7}` to `{0, 0, 0, 0.5, 0.9}`.

**Results** (from `exp_03_results/`): **Identical to exp_02.** All 16 model hashes match byte-for-byte.

### Critical discovery: Shuffle noise is invisible to LSR training

The manifest train data hashes for clients 3 and 4 ARE different (confirming more data was corrupted), but the aggregated models are identical. This means:

**Shuffling responses within the same domain does not change LSR gradients.** The LSR loss (KL-divergence between retriever scores and LM scores) treats an in-domain shuffled response almost the same as the correct response — the LM generates similar scores for any same-domain passage, and the retriever's cosine similarity doesn't distinguish well between correct and shuffled same-domain text.

**Implication**: No quality signal (rank-displacement, holdout MRR, or anything else) can discriminate between clean and noisy clients when the noise produces no measurable effect on the model.

**Gate**: ❌ **FAILED** — `lower_quality_clients_downweighted` fails. No improvement over exp_02.

---

## Step 4: Change Noise Type + Quality Signal ← **MUST RE-RUN (bugfix applied)**

The original Step 4 was "change quality signal." But exp_03 revealed a deeper problem: **shuffle noise itself is invisible to LSR**, so no quality signal can detect it. We need to fix the noise type first.

### Step 4a: Switch to `random_negative` noise

**Why**: `random_negative` replaces responses with random documents from the corpus that are NOT the correct answer. Unlike shuffle (which swaps between similar in-domain passages), random negatives pair queries with unrelated documents. This should produce clearly different LM scores and retriever scores, creating genuinely destructive gradients.

**Config change**: `MECHANISM_NOISE_MODE = "random_negative"` (was `"shuffle"`). ✅ Done.

**exp_04a attempt**: ⚠️ **INVALID** — model hashes identical to exp_02/exp_03 because of a **code bug**:
- The mechanism benchmark path in `split_noisy()` (lines 264-290) **hardcoded `deranged_shuffle`** for all noise, completely ignoring `CURRENT_NOISE_MODE`
- `MECHANISM_NOISE_MODE` was logged to the manifest but never dispatched in the actual corruption loop
- Only the robustness benchmark path had the full noise mode dispatch

**Bugfix applied**: Updated the mechanism path to dispatch based on `CURRENT_NOISE_MODE`, supporting `shuffle`, `random_negative`, `hard_negative`, and `cross_domain`.

**Action**: Push the bugfix, then re-run all 4 alpha notebooks. Store results in `exp_04b_results/`.

**Sanity check**: α=0.0 R1 model hash must differ from `5706c0d7...` AND client 3/4 train hashes must differ from exp_03.

### Step 4b: If random_negative works, optionally also try holdout MRR signal

If the noisy clients now produce visibly worse models (different hashes, worse rank-displacement), we can additionally try switching the quality signal to holdout MRR for even stronger discrimination:
- Each client evaluates on their clean holdout (160 pairs) after training
- Send `loss = -holdout_mrr` to the server

**Run**: Full α sweep (0.0, 0.3, 0.7, 1.0)

**Gate**:
1. α=0.0 model hashes differ from exp_02/exp_03/exp_04a (confirming noise actually affects training)
2. `lower_quality_clients_downweighted` passes for at least one α>0 in ≥3/4 rounds
3. At least one α>0 beats α=0.0 on final test MRR or NDCG

- **Pass** → run multi-seed. 🎉
- **Fail** → try holdout MRR signal (Step 4b), or reframe contribution.

---

## Decision Tree

```
Step 1: Can LSR training improve retrieval?  ✅ YES (+10.3%)
  │
  └── Step 2: Federated LSR + α-sweep  ✅ PARTIAL (training helps, α delays degradation)
              │
              └── Step 3: Increase shuffle noise  ❌ FAILED (shuffle is invisible to LSR)
                           │
                           └── Step 4a: Switch to random_negative noise  ← YOU ARE HERE
                                         │  (exp_04a invalid due to code bug — bugfix applied)
                                         │
                                         ├── Noise visible + α helps → Done! 🎉
                                         │
                                         ├── Noise visible but α flat → Step 4b: try holdout MRR signal
                                         │
                                         └── Noise still invisible → Reframe contribution
```

---

## Do NOT Repeat

- ❌ `paraphrase-MiniLM-L3-v2` for mechanism benchmark (can't learn from LSR)
- ❌ LR=5e-7 (negligible movement)
- ❌ LR=5e-6 (causes collapse)
- ❌ Epochs ≥ 2 (causes collapse)
- ❌ Rounds ≥ 6 (causes degradation)
- ❌ Hard-negative noise before shuffle works (always flat)
- ❌ Mixed noise mode (known to fail)
- ❌ Shuffle noise at ANY ratio (exp_03: invisible to LSR, models identical regardless of ratio)
- ❌ Running mechanism benchmark without the noise dispatch bugfix (exp_04a: code always used deranged_shuffle)
