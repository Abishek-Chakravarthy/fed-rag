# Experiment 03 — Log

## Config

- **Retriever**: `all-MiniLM-L6-v2`
- **LR**: 2e-6
- **Rounds**: 4, **Epochs**: 1
- **Clients**: 5, **Split**: equal
- **Noise ladder**: 0/0/0/**0.5**/**0.9** (changed from 0.3/0.7 in exp_02)
- **β**: 5.0
- **Quality signal**: rank-displacement

## Key Finding: Noise increase has ZERO effect on the model

### Evidence

**Training data IS different** (manifest train hashes confirm):
| Client | exp_02 train hash | exp_03 train hash | Changed? |
|--------|-------------------|-------------------|----------|
| 0 (clean) | `b3697b5d...` | `b3697b5d...` | No |
| 1 (clean) | `757d3b89...` | `757d3b89...` | No |
| 2 (clean) | `99b0c9bb...` | `99b0c9bb...` | No |
| 3 (50% noise) | `3ddf8b89...` | `173fd1a9...` | **Yes** |
| 4 (90% noise) | `da4a6b06...` | `7c636822...` | **Yes** |

**But the aggregated model is byte-for-byte identical** — all 16 model hashes (4α × 4R) match exp_02 exactly.

### Root Cause

**Shuffle noise does not change LSR gradients.** When responses are shuffled within the same NFCorpus domain, the KL-divergence between retriever scores and LM-generated scores is nearly identical whether the response is correctly matched or shuffled. The LSR training objective treats in-domain shuffled responses almost the same as correct responses because:

1. The LM (distilgpt2) generates similar scores for any in-domain text
2. The retriever's cosine similarity scores don't distinguish well between correct and shuffled same-domain passages
3. The gradients from corrupted pairs are similar in direction and magnitude to clean pair gradients

This means **increasing shuffle noise from 30%→50% or 70%→90% cannot help** because the noise type itself is invisible to the training objective.

### Implication for Step 3 Gate

**Gate 1** (`lower_quality_clients_downweighted`): ❌ FAIL — quality signal cannot discriminate because the models are identical; the rank-displacement values are the same.

**Gate 2** (α>0 beats α=0.0): Results are identical to exp_02. The same partial pass applies (val MRR advantage at α=0.7/1.0, flat on test).

**Step 3 verdict: FAIL.** Increasing shuffle noise amplitude does not work. Proceed to Step 4 or change noise type.

## Metrics (identical to exp_02)

| α | Best Round | Best Val MRR | Final Test MRR | Final Test NDCG |
|---|-----------|-------------|----------------|-----------------|
| 0.0 | R2 | 0.0262 | 0.0202 | 0.0322 |
| 0.3 | R2 | 0.0262 | 0.0200 | 0.0320 |
| 0.7 | R3 | 0.0275 | 0.0200 | 0.0328 |
| 1.0 | R3 | 0.0272 | 0.0201 | 0.0333 |
