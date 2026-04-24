# Experiment 04a — Log

## Config (intended)

- **Retriever**: `all-MiniLM-L6-v2`
- **LR**: 2e-6
- **Rounds**: 4, **Epochs**: 1
- **Clients**: 5, **Split**: equal
- **Noise ladder**: 0/0/0/0.5/0.9
- **Noise mode**: `random_negative` (changed from `shuffle`)
- **β**: 5.0
- **Quality signal**: rank-displacement

## ⚠️ CRITICAL FINDING: Noise mode change had NO effect — CODE BUG

### Evidence

All 16 model hashes (4α × 4R) are **identical to exp_02 and exp_03**. The training data hashes for clients 3 and 4 are also identical to exp_03.

| Metric | exp_02 (shuffle 0.3/0.7) | exp_03 (shuffle 0.5/0.9) | exp_04a (random_neg 0.5/0.9) |
|--------|--------------------------|--------------------------|------------------------------|
| α=0.0 R1 hash | `5706c0d7...` | `5706c0d7...` | `5706c0d7...` |
| Client 3 train hash | `3ddf8b89...` | `173fd1a9...` | `173fd1a9...` (= exp_03!) |
| Client 4 train hash | `da4a6b06...` | `7c636822...` | `7c636822...` (= exp_03!) |

### Root cause: Bug in `split_noisy()` mechanism path

The mechanism benchmark code path (lines 264-290 of `federated_noisy_qa.py`) **hardcoded `deranged_shuffle`** for all noise corruption, completely ignoring `CURRENT_NOISE_MODE`:

```python
# OLD BUG: line 273 — always shuffles regardless of noise mode
replacement_responses = deranged_shuffle(original_responses, rng)
```

The `MECHANISM_NOISE_MODE` config was logged to the manifest but never actually dispatched in the data corruption loop. Only the robustness benchmark path (lines 291+) had the full noise mode dispatch.

### Fix applied

Updated the mechanism path to dispatch based on `CURRENT_NOISE_MODE`, supporting `shuffle`, `random_negative`, `hard_negative`, and `cross_domain`.

## Metrics (identical to exp_02/exp_03 — invalid for Step 4a evaluation)

| α | Best Round | Best Val MRR | Final Test MRR | Final Test NDCG |
|---|-----------|-------------|----------------|-----------------|
| 0.0 | R2 | 0.0262 | 0.0202 | 0.0322 |
| 0.3 | R2 | 0.0262 | 0.0200 | 0.0320 |
| 0.7 | R3 | 0.0275 | 0.0200 | 0.0328 |
| 1.0 | R3 | 0.0272 | 0.0201 | 0.0333 |

**Step 4a has NOT been validly executed.** Must re-run after pushing the bugfix.
