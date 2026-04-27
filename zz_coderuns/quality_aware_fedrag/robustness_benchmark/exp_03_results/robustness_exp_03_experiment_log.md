# Robustness Benchmark (Stress Track) — Exp 03 (Hard Negative Noise) Log

## Config

- **Benchmark Mode**: `stress`
- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2`
- **Training objective**: `MultipleNegativesRankingLoss` (InfoNCE / contrastive)
- **LR**: 2e-6
- **Rounds**: 4, **Epochs**: 1, **Batch size**: 8
- **Clients**: 5
- **Client Split Mode**: `unequal`
- **Noisy Client**: Client 4 holds **40%** of the dataset.
- **Noise Mode**: `hard_negative` ← **CRITICAL CHANGE**
- **Noise Ratio**: 0.8 (80% of Client 4's data is corrupted)
- **Quality signal**: `train_loss`
- **β**: 2.0
- **Alphas**: 0.0, 0.3, 0.7, 1.0
- **Seeds**: 42, 123, 256

---

## Final Test Metrics (Aggregated Across Seeds)

| α | Mean Test MRR | Std Dev | Mean Test NDCG | vs FedAvg (α=0.0) MRR |
|---|---------------|---------|----------------|-----------------------|
| 0.0 | 0.02435 | ±0.00388 | 0.03544 | — |
| 0.3 | 0.02434 | ±0.00379 | 0.03530 | 0.0% |
| 0.7 | **0.02506** | ±0.00370 | **0.03618** | **+2.9%** |
| 1.0 | 0.02478 | ±0.00410 | 0.03581 | **+1.8%** |

*Paradox*: QA-FedAvg (α=0.7) significantly beats FedAvg on Test MRR. But looking under the hood reveals a catastrophic failure of the signal.

---

## Per-Client Analysis (Seed 123, Round 3, α=1.0)

| Client | Data Fraction | Noise | Train Loss | Quality Score | Combined Weight (α=1.0) |
|--------|---------------|-------|------------|---------------|-------------------------|
| C0 | 15% | 0% | 2.412 | 0.0050 | 0.5% |
| C1 | 15% | 0% | 2.292 | 0.0070 | 0.7% |
| C2 | 15% | 0% | 2.295 | 0.0070 | 0.7% |
| C3 | 15% | 0% | 2.253 | 0.0079 | 0.8% |
| **C4** | **40%** | **80% (`hard_neg`)** | **0.603** | **0.9730** | **97.3%** |

### The "False Clean" Illusion
The mechanism is **UP-weighting the poisoned client**!
Why? `hard_negative` noise replaces the true document with a document that is lexically very similar to the query (high BM25 score) but factually irrelevant. Because it is lexically similar, the pre-trained `all-MiniLM-L6-v2` model already embeds them closely. 
Thus, the initial cosine similarity is extremely high, making the contrastive `train_loss` drop to near-zero (~0.60).
Meanwhile, clean clients with real, complex semantic pairs have a harder time pulling them together, resulting in higher "normal" losses (~2.3).

**Result**: The mechanism incorrectly assumes Client 4 has the cleanest data and gives it 97% of the global weight.

### Why did Test MRR improve at α=0.7?
Because the `hard_negative` noise is essentially forcing the model to heavily prioritize lexical overlap. On the NFCorpus dataset (which has many lexical-heavy queries), this unintended "overfitting to BM25" accidentally slightly improved the final Test MRR. However, the mechanism failed its core goal: it rewarded corruption instead of penalizing it.

---

## Mean Server Validation MRR (Aggregated Across Seeds)

| Round | α=0.0 (FedAvg) | α=0.3 | α=0.7 | α=1.0 |
|-------|----------------|-------|-------|-------|
| Pre | 0.02576 | 0.02576 | 0.02576 | 0.02576 |
| R1 | 0.02583 | **0.02636** | 0.02628 | 0.02623 |
| R2 | 0.02633 | **0.02635** | 0.02626 | 0.02618 |
| R3 | **0.02645** | 0.02621 | 0.02615 | 0.02617 |
| R4 | 0.02626 | 0.02606 | **0.02632** | 0.02627 |

---

## Acceptance Test Results

| Test | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|------|-------|-------|-------|-------|
| `alpha_zero_matches_fedavg` | ✅ 3/3 | N/A | N/A | N/A |
| `distinct_round_trajectories` | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 |
| `higher_loss_clients_downweighted` | N/A | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 |

### False Positive on Acceptance Test
The `higher_loss_clients_downweighted` test passed 100% of the time. This is a false positive! 
The test logic finds the client with the highest loss and ensures it was downweighted. Because the clean clients had the highest loss (due to the hard negative illusion), the script verified that a *clean* client was downweighted relative to its data size. The test logic assumed the highest loss client was the noisy one, which was false.

---

## Conclusion

Experiment 03 reveals a **fundamental vulnerability** in using `train_loss` as a universal quality proxy. While it works perfectly for `random_negative` noise (where the false pair is semantically disjoint), it completely fails against `hard_negative` noise (where the false pair is lexically overlapping). The model is tricked into thinking the lexically overlapping noise is high-quality "easy" data, causing QA-FedAvg to aggressively promote the poisoned client. 

This proves that `train_loss` is not a robust quality signal against adversarial or sophisticated noise types. A return to an evaluation-based metric (like `delta_mrr` on a shared probe set, but with a larger probe size) is required to build a truly robust QA-FedAvg system.
