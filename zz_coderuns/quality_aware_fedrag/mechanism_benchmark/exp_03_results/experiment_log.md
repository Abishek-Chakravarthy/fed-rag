# Experiment 03 — Log

## Config (intended)

- **Retriever**: `all-MiniLM-L6-v2`
- **LR**: 2e-6
- **Rounds**: 4, **Epochs**: 1
- **Clients**: 5, **Split**: equal
- **Noise ladder**: 0/0/0/**0.5**/**0.9** (changed from 0.3/0.7)
- **β**: 5.0
- **Quality signal**: rank-displacement

## ⚠️ CRITICAL FINDING: Results are identical to exp_02

**Every single model hash across all 4 alphas and all 4 rounds is byte-for-byte identical to exp_02** (where noise was 0.3/0.7).

Evidence:
- α=0.0 R1 hash: `5706c0d7...` (same in exp_02 and exp_03)
- α=0.0 R2 hash: `c5ef46c6...` (same)
- α=1.0 R4 hash: `56a4b74f...` (same)
- All 16 hashes (4 alphas × 4 rounds) are identical

The manifests correctly log `noise_3=0.50, noise_4=0.90`, but the actual training used the old 0.3/0.7 values.

**Root cause**: The noise change was committed locally after the Kaggle notebooks were launched. The Kaggle notebooks cloned an older version of `q-fedrag2` that still had 0.3/0.7.

**Action required**: Re-run the Kaggle notebooks after confirming the latest commit (`23a9643`) with the 0.5/0.9 noise values is on the remote branch.

## Metrics (identical to exp_02, not valid for Step 3 evaluation)

| α | Best Round | Best Val MRR | Final Test MRR | Final Test NDCG |
|---|-----------|-------------|----------------|-----------------|
| 0.0 | R2 | 0.0262 | 0.0202 | 0.0322 |
| 0.3 | R2 | 0.0262 | 0.0200 | 0.0320 |
| 0.7 | R3 | 0.0275 | 0.0200 | 0.0328 |
| 1.0 | R3 | 0.0272 | 0.0201 | 0.0333 |

These are exp_02 results re-run. Step 3 has NOT been executed yet.
