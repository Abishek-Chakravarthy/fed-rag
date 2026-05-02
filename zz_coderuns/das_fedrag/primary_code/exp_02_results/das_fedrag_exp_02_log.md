# DAS-FedAvg Experiment 02 Log

**Experiment Name**: `das_fedavg_soft_domain_weighting` (8 Rounds, Deterministic Eval Splits)
**Target Dataset**: `nfcorpus` (medical domain)

## 1. Experiment Settings
- **Rounds**: 8
- **Local Epochs**: 1
- **Learning Rate**: 2e-06
- **Batch Size**: 8
- **Number of Clients**: 5
- **Max Corpus Docs**: 8000
- **Max Train Pairs**: 4000
- **Max Server Val Pairs**: 400
- **Max Final Test Pairs**: 500
- **Top K**: 10

## 2. Client Split Summary & Relevance Scores
| Client ID | Domain | Relevance Score | Num Training Pairs | Pair Hash |
| :--- | :--- | :--- | :--- | :--- |
| 0 | medical (TARGET) | 1.0000 | 4000 | varies by seed |
| 1 | science | 0.4791 | 239 | varies by seed |
| 4 | biomedical | 0.3212 | 101 | varies by seed |
| 3 | argument | 0.0422 | 981 | varies by seed |
| 2 | finance | 0.0130 | 168 | varies by seed |

## 3. Universal Pre-Train Metrics (Deterministic Eval Splits)
Due to the implementation of deterministic validation and test splits (independent of the training data shuffle seed), the pre-train metrics are now completely identical across all runs and algorithms. This provides a unified baseline for fair comparison.

- **Pre-Server Val**: MRR: 0.0224 | Recall@10: 0.0575 | NDCG@10: 0.0304
- **Pre-Final Test**: MRR: 0.0272 | Recall@10: 0.0800 | NDCG@10: 0.0392

---

## 4. Evaluation Metrics Summary

### Baseline FedAvg (Seed 42)
- **Best Server Val**: MRR: 0.0224 | Recall@10: 0.0625 | NDCG@10: 0.0314
- **Final Test**: MRR: 0.0255 | Recall@10: 0.0800 | NDCG@10: 0.0379
- **Best Round**: 8

### Soft Domain Weighting (Seed 42)
- **Best Server Val**: MRR: 0.0225 | Recall@10: 0.0625 | NDCG@10: 0.0316
- **Final Test**: MRR: 0.0252 | Recall@10: 0.0760 | NDCG@10: 0.0367
- **Best Round**: 8

### Baseline FedAvg (Seed 123)
- **Best Server Val**: MRR: 0.0224 | Recall@10: 0.0600 | NDCG@10: 0.0310
- **Final Test**: MRR: 0.0262 | Recall@10: 0.0760 | NDCG@10: 0.0376
- **Best Round**: 2

### Soft Domain Weighting (Seed 123)
- **Best Server Val**: MRR: 0.0222 | Recall@10: 0.0600 | NDCG@10: 0.0308
- **Final Test**: MRR: 0.0261 | Recall@10: 0.0760 | NDCG@10: 0.0376
- **Best Round**: 2

### Baseline FedAvg (Seed 256)
- **Best Server Val**: MRR: 0.0227 | Recall@10: 0.0625 | NDCG@10: 0.0317
- **Final Test**: MRR: 0.0279 | Recall@10: 0.0800 | NDCG@10: 0.0398
- **Best Round**: 2

### Soft Domain Weighting (Seed 256)
- **Best Server Val**: MRR: 0.0234 | Recall@10: 0.0600 | NDCG@10: 0.0317
- **Final Test**: MRR: 0.0269 | Recall@10: 0.0780 | NDCG@10: 0.0386
- **Best Round**: 4

---

## 5. Final Comparison Table (Final Test Metrics)

| Model | Seed | Final Test MRR | Final Test Recall@10 | Final Test NDCG@10 | Best Round |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline FedAvg** | 42 | 0.0255 | 0.0800 | 0.0379 | 8 |
| **Soft Domain** | 42 | 0.0252 | 0.0760 | 0.0367 | 8 |
| **Baseline FedAvg** | 123 | 0.0262 | 0.0760 | 0.0376 | 2 |
| **Soft Domain** | 123 | 0.0261 | 0.0760 | 0.0376 | 2 |
| **Baseline FedAvg** | 256 | 0.0279 | 0.0800 | 0.0398 | 2 |
| **Soft Domain** | 256 | 0.0269 | 0.0780 | 0.0386 | 4 |

## 6. Critical Observations
1. **Eval Splits are Fixed**: Pre-train MRR is universally `0.0272` (Final Test) across all seeds and models. The fix to decouple the eval split from the shuffle seed was successful.
2. **Negative Performance Signal**: Across all three seeds, the Soft Domain Weighting mechanism slightly **underperforms** the Baseline FedAvg on the target domain `nfcorpus`. 
   - Seed 42: Baseline MRR 0.0255 > Soft Domain MRR 0.0252
   - Seed 123: Baseline MRR 0.0262 > Soft Domain MRR 0.0261
   - Seed 256: Baseline MRR 0.0279 > Soft Domain MRR 0.0269
3. **Training Degradation**: Even at their best rounds, both Baseline and Soft Domain models perform worse than the untrained initial model (Pre-Train Final Test MRR = `0.0272`). For example, Baseline Seed 123 best round MRR is `0.0262`, indicating that the federated training is largely destructive against this specific evaluation set.
4. **Hypothesis**: The combination of `nfcorpus` with the specific contrastive training setup might not be yielding positive generalization. Since Soft Domain restricts updates primarily to the `nfcorpus` client, and the training is destructive, the Soft Domain model degrades slightly faster/worse than Baseline FedAvg which diffuses the destructive updates across more domains.
