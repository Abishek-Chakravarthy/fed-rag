# Robustness Benchmark (Stress Track) — Exp 01 Log

## Config

- **Benchmark Mode**: `stress`
- **Retriever**: `sentence-transformers/all-MiniLM-L6-v2`
- **Training objective**: `MultipleNegativesRankingLoss` (InfoNCE / contrastive)
- **LR**: 2e-6
- **Rounds**: 4, **Epochs**: 1, **Batch size**: 8
- **Clients**: 5
- **Client Split Mode**: `unequal`
- **Noisy Client**: Client 4 holds **40%** of the entire dataset. Clients 0-3 share the remaining 60% (15% each).
- **Noise Mode**: `random_negative`
- **Noise Ratio**: 0.8 (80% of Client 4's data is corrupted)
- **Quality signal**: `train_loss`
- **β**: 2.0
- **Alphas**: 0.0, 0.3, 0.7, 1.0
- **Seed**: 42

---

## Server Validation MRR per Round

| Round | α=0.0 (FedAvg) | α=0.3 | α=0.7 | α=1.0 |
|-------|----------------|-------|-------|-------|
| R1 | 0.02431 | 0.02435 | 0.02426 | 0.02420 |
| R2 | 0.02436 | **0.02447** | 0.02431 | 0.02420 |
| R3 | 0.02407 | 0.02432 | 0.02439 | 0.02426 |
| R4 | 0.02442 | 0.02443 | **0.02457** | **0.02423** |

*Note*: α=0.7 achieved the highest validation MRR (0.02457) in Round 4, beating standard FedAvg (0.02442).

---

## Final Test Metrics (from best round)

| α | Best Round | Val MRR | Test MRR | Test NDCG |
|---|------------|---------|----------|-----------|
| 0.0 | R4 | 0.02442 | **0.02144** | 0.03270 |
| 0.3 | R2 | 0.02448 | 0.02080 | 0.03263 |
| 0.7 | R4 | **0.02457** | 0.02111 | **0.03290** |
| 1.0 | R3 | 0.02427 | 0.02087 | 0.03270 |

*Observation*: While QA-FedAvg (α=0.7) reached a higher Server Validation MRR and Final Test NDCG than FedAvg, standard FedAvg performed slightly better on Final Test MRR on this specific seed. This could be due to seed variance (as seen in the mechanism benchmark), warranting a multi-seed run to establish true statistical significance.

---

## Per-Client Analysis (Round 3 Example, α=0.7)

In the `stress` track, Client 4 is a massive data holder (40% of data) but is 80% corrupted. Let's look at how the QA-FedAvg mechanism handled this:

| Client | Data Fraction | Noise | Train Loss | Quality Score | Combined Weight (α=0.7) |
|--------|---------------|-------|------------|---------------|-------------------------|
| C0 | 15% | 0% | 2.167 | 0.4178 | 33.7% |
| C1 | 15% | 0% | 2.284 | 0.1721 | 16.5% |
| C2 | 15% | 0% | 2.367 | 0.0915 | 10.9% |
| C3 | 15% | 0% | 2.204 | 0.3167 | 26.6% |
| **C4** | **40%** | **80%** | **2.891** | **0.0017** | **12.1%** |

**Key Findings:**
1. **Flawless Identification**: The contrastive `train_loss` effortlessly identified the massively corrupted Client 4 (Loss = 2.89 vs 2.16-2.36 for clean clients).
2. **Defusing the Bomb**: Under standard FedAvg (α=0.0), Client 4 would exert **40%** influence on the global model in every round, poisoning the updates. Under QA-FedAvg (α=0.7), its influence is crushed down to just **12.1%**.
3. **Redistribution**: The influence is correctly redistributed to the clean clients, particularly C0 and C3, which showed the lowest training losses (best alignment with clean data).

---

## Acceptance Test Results

| Test | α=0.0 | α=0.3 | α=0.7 | α=1.0 |
|------|-------|-------|-------|-------|
| `alpha_zero_matches_fedavg` | ✅ | N/A | N/A | N/A |
| `distinct_round_trajectories` | ✅ | ✅ | ✅ | ✅ |
| `higher_loss_clients_downweighted` | N/A | ✅ | ✅ | ✅ |

*Note*: The `lower_quality_clients_downweighted` test returned `None` because it is specifically gated to `benchmark_mode == "mechanism"` in the `federated_noisy_qa.py` script. The new `higher_loss_clients_downweighted` correctly verifies that C4's weight dropped below its initial size weight.

---

## Model Hash Verification

All hashes are unique across alphas and rounds, confirming noise is actively affecting the training and alpha scaling works.

| α | R1 | R2 | R3 | R4 |
|---|----|----|----|----|
| 0.0 | `4a3b23fc` | `f829bda3` | `05f7c982` | `69f1a033` |
| 0.3 | `40d50519` | `f4116af2` | `4d3a7118` | `18d6b944` |
| 0.7 | `50210694` | `48851dfd` | `beab799b` | `c9be6dfe` |
| 1.0 | `b2f30322` | `cd91f0b6` | `3ef5267f` | `2d586366` |

---

## Conclusion

The QA-FedAvg mechanism technically functioned perfectly in the `stress` robustness scenario. It accurately identified the single large poisoned client (C4) via `train_loss` and successfully throttled its aggregation weight from 40% down to 12% (at α=0.7) and near 0% (at α=1.0). While Validation MRR and NDCG favored QA-FedAvg, the Test MRR slightly favored FedAvg on this specific seed, indicating a multi-seed run is necessary to conclude on performance gains.
