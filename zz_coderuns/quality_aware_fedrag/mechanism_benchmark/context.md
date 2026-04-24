# QA-FedRAG Mechanism Benchmark — Full Context

> **Purpose**: Give a new coding agent everything it needs to resume this work from scratch.

---

## Project Overview

This is a **final year project** implementing **Quality-Aware Federated Retrieval-Augmented Generation (QA-FedRAG)**. The core contribution is a modified FedAvg algorithm that weights client model updates based on data quality, downweighting noisy/low-quality clients.

### QA-FedAvg Formula

Standard FedAvg: `w_global = Σ (n_i / n_total) * w_i`

QA-FedAvg: `w_global = Σ weight_i * w_i`

Where `weight_i = softmax(-β * loss_i)` blended with size weight:
```
quality_score_i = softmax(-β * loss_i)          # β = 5.0
weight_i = (1 - α) * size_weight_i + α * quality_score_i
```

- `α = 0.0` → pure FedAvg (equal weights)
- `α = 1.0` → pure quality-based weighting
- `loss_i` = per-client quality signal (currently: shared rank displacement)
- `β` = temperature parameter controlling how aggressively bad clients are penalized

### Repository Structure

```
fed-rag/
├── src/fed_rag/                          # Library code
│   └── trainers/huggingface/lsr.py       # LSR trainer (has DataParallel fix)
└── zz_coderuns/quality_aware_fedrag/
    └── mechanism_benchmark/              # <-- All experiment code lives here
        ├── federated_noisy_qa.py         # Main experiment script (1600 lines)
        ├── prepare_beir_data.py          # Data loading from NFCorpus/BEIR
        ├── single_alpha_kaggle_colab.ipynb  # Kaggle notebook to run experiments
        ├── improvement_plan_from_here.md # Master plan (READ THIS FIRST)
        ├── context.md                    # This file
        ├── lsr_training_check/           # Step 1 diagnostic results
        ├── exp_02_results/               # Step 2 results (shuffle 0.3/0.7)
        ├── exp_03_results/               # Step 3 results (shuffle 0.5/0.9 — identical to exp_02!)
        └── exp_04a_results/              # Step 4a results (random_neg — bug, still used shuffle)
```

### Execution Environment

- **Training**: Kaggle notebooks with dual T4 GPUs
- **Branch**: `q-fedrag2` on GitHub (`Abishek-Chakravarthy/fed-rag`)
- The notebook clones the branch, installs deps, runs `federated_noisy_qa.py` as subprocess
- Each alpha value (0.0, 0.3, 0.7, 1.0) runs in a separate notebook copy

---

## Current State (as of April 24, 2026)

### What works
- LSR training with `all-MiniLM-L6-v2` retriever at lr=2e-6 **does improve** retrieval (+10.3% MRR in diagnostic)
- The QA-FedAvg weighting mechanism is mathematically correct
- α>0 delays model degradation by ~1 federated round vs α=0.0 (val MRR advantage)
- `DataParallel` unwrapping fix in `lsr.py` works for multi-GPU Kaggle

### What's broken / discovered
1. **Shuffle noise is invisible to LSR** — exp_03 proved that changing shuffle ratio from 0.3/0.7 to 0.5/0.9 produces byte-for-byte identical models. The LSR loss treats in-domain shuffled responses the same as correct ones.
2. **Mechanism path had a bug** — `split_noisy()` hardcoded `deranged_shuffle` for mechanism mode, ignoring `MECHANISM_NOISE_MODE`. Fixed but not yet tested.
3. **Notebook slug mismatch** — Notebook Cell 1 has `NOISE_MODE = "shuffle"` but script uses `MECHANISM_NOISE_MODE = "random_negative"` for mechanism mode. The export cell will FileNotFoundError. **Must change notebook Cell 1 to `NOISE_MODE = "random_negative"`.**

### Immediate Next Step

**Push bugfix + re-run Step 4a with `random_negative` noise:**
1. Commit the noise dispatch fix in `federated_noisy_qa.py`
2. Push to `q-fedrag2`
3. In Kaggle notebook Cell 1, change `NOISE_MODE = "random_negative"`
4. Run 4 notebooks (α=0.0, 0.3, 0.7, 1.0)
5. Download results → `exp_04b_results/`
6. Sanity check: α=0.0 R1 model hash must differ from `5706c0d7...`

---

## Key Architectural Details

### How mechanism benchmark noise works

`federated_noisy_qa.py` has two benchmark modes:
- **mechanism**: 5 clients with per-client noise ladder (e.g., 0/0/0/0.5/0.9). Tests whether α>0 can downweight noisy clients.
- **stress** (robustness): 1 noisy client vs 4 clean. Tests resilience under attack.

For mechanism mode, `resolve_benchmark_config()` (line 1109) **ignores CLI args** and uses `MECHANISM_*` constants:
```python
MECHANISM_NOISE_MODE = "random_negative"  # line 73
MECHANISM_CLIENT_NOISE_MAP = {"0":0, "1":0, "2":0, "3":0.5, "4":0.9}  # lines 66-72
MECHANISM_CLIENT_SPLIT_MODE = "equal"  # line 74
MECHANISM_RETRIEVER = "all-MiniLM-L6-v2"  # line 75
```

### Quality signal flow

1. Each client trains locally on their (possibly corrupted) data
2. After training, each client's model is evaluated on a **shared quality set** (400 pairs)
3. The rank displacement (how much each client's update moved correct documents in ranking) is computed
4. `loss_i = mean(max(rank_after - rank_before, 0)) - 0.5 * mean(max(rank_before - rank_after, 0))`
5. Server computes `quality_score_i = softmax(-β * loss_i)`
6. Aggregation weights: `w_i = (1-α) * size_weight + α * quality_score`

### Model hash tracking

Every round, the aggregated model's state_dict is hashed (MD5). This is the primary tool for detecting whether noise actually affects training. If hashes match across experiments, the models are byte-for-byte identical.

---

## Experiment History (chronological)

| Exp | Config Change | Result | Key Finding |
|-----|--------------|--------|-------------|
| Diag | Single-client LSR, no federation | `paraphrase-MiniLM-L3-v2` degrades, `all-MiniLM-L6-v2` improves +10.3% | Retriever choice is critical |
| exp_02 | Switch to `all-MiniLM-L6-v2`, shuffle 0.3/0.7 | Training works; α>0 peaks 1 round later | Quality signal doesn't discriminate |
| exp_03 | Increase shuffle to 0.5/0.9 | **Identical** model hashes to exp_02 | Shuffle noise invisible to LSR |
| exp_04a | Set `MECHANISM_NOISE_MODE="random_negative"` | **Identical** hashes again | Code bug: mechanism path ignored noise mode |
| exp_04b | Bugfix: dispatch noise mode in mechanism path | **Not yet run** | Pending |

---

## Known Constraints (DO NOT REPEAT)

- ❌ `paraphrase-MiniLM-L3-v2` — can't learn from LSR
- ❌ LR=5e-7 — negligible movement
- ❌ LR=5e-6 — causes collapse
- ❌ Epochs ≥ 2 — causes collapse
- ❌ Rounds ≥ 6 — causes degradation
- ❌ Shuffle noise at any ratio — invisible to LSR
- ❌ Running mechanism benchmark without noise dispatch fix

---

## Files to Read (priority order)

1. **This file** (`context.md`) — You're reading it
2. **`improvement_plan_from_here.md`** — Master plan with decision tree and gates
3. **`federated_noisy_qa.py`** — Main script, focus on:
   - Lines 64-76: `MECHANISM_*` constants
   - Lines 264-314: `split_noisy()` mechanism noise dispatch (recently fixed)
   - Lines 1100-1130: `resolve_benchmark_config()`
   - Lines 1133-1240: `main()` function
   - Lines 1530-1602: CLI argument parser
4. **`single_alpha_kaggle_colab.ipynb`** — Kaggle execution notebook
5. **`exp_03_results/experiment_log.md`** — Shuffle invisibility proof
6. **`exp_04a_results/experiment_log.md`** — Code bug documentation
7. **`lsr_training_check/diagnostic_summary.json`** — Retriever diagnostic
8. **`src/fed_rag/trainers/huggingface/lsr.py`** — LSR trainer (DataParallel fix at line ~95)
9. **`prepare_beir_data.py`** — Data loading (retriever model config at top)

---

## Critical Code Locations

| What | File | Lines | Why it matters |
|------|------|-------|---------------|
| Mechanism config constants | `federated_noisy_qa.py` | 64-76 | All noise/retriever/split settings for mechanism mode |
| Noise dispatch (FIXED) | `federated_noisy_qa.py` | 264-314 | Was hardcoded to shuffle; now dispatches on mode |
| Config resolution | `federated_noisy_qa.py` | 1100-1130 | Mechanism mode ignores CLI, uses MECHANISM_* |
| Run slug builder | `federated_noisy_qa.py` | 531-547 | Must match notebook slug for export |
| Quality signal computation | `federated_noisy_qa.py` | ~1350-1420 | Rank displacement + softmax weighting |
| DataParallel fix | `lsr.py` | ~92-110 | Unwraps module for multi-GPU |
| Notebook config | `single_alpha_kaggle_colab.ipynb` | Cell 1 | NOISE_MODE must match MECHANISM_NOISE_MODE |
