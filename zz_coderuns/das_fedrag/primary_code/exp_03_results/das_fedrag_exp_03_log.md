# DAS-FedAvg Experiment 03 Log

**Experiment Name**: `das_fedavg_soft_domain_weighting` (8 Rounds, LR=5e-7, Temperature=1.0)
**Target Dataset**: `nfcorpus` (medical domain)

## 1. Experiment Settings
- **Rounds**: 8
- **Local Epochs**: 1
- **Learning Rate**: 5e-07 (Reduced from 2e-06 to prevent destructive training)
- **Batch Size**: 8
- **Number of Clients**: 5
- **Max Corpus Docs**: 8000
- **Max Train Pairs**: 4000
- **Max Server Val Pairs**: 400
- **Max Final Test Pairs**: 500
- **Top K**: 10
- **Domain Weighting**: Temperature-scaled Softmax (τ = 1.0)

## 2. Client Split Summary & Relevance Scores
| Client ID | Domain | Relevance Score | Num Training Pairs | Pair Hash |
| :--- | :--- | :--- | :--- | :--- |
| 0 | medical (TARGET) | 1.0000 | 4000 | varies by seed |
| 1 | science | 0.4791 | 239 | varies by seed |
| 4 | biomedical | 0.3212 | 101 | varies by seed |
| 3 | argument | 0.0422 | 981 | varies by seed |
| 2 | finance | 0.0130 | 168 | varies by seed |

## 3. Universal Pre-Train Metrics (Deterministic Eval Splits)
- **Pre-Server Val**: MRR: 0.0224 | Recall@10: 0.0575 | NDCG@10: 0.0304
- **Pre-Final Test**: MRR: 0.0272 | Recall@10: 0.0800 | NDCG@10: 0.0392

---

## 4. Evaluation Metrics Summary

### Baseline FedAvg (Seed 42)
- **Best Server Val**: MRR: 0.0222 | Recall@10: 0.0600 | NDCG@10: 0.0308
- **Final Test**: MRR: 0.0271 | Recall@10: 0.0780 | NDCG@10: 0.0387
- **Best Round**: 6

### Soft Domain Weighting (Seed 42)
- **Best Server Val**: MRR: 0.0224 | Recall@10: 0.0650 | NDCG@10: 0.0320
- **Final Test**: MRR: 0.0259 | Recall@10: 0.0780 | NDCG@10: 0.0378
- **Best Round**: 8

### Baseline FedAvg (Seed 123)
- **Best Server Val**: MRR: 0.0223 | Recall@10: 0.0600 | NDCG@10: 0.0309
- **Final Test**: MRR: 0.0262 | Recall@10: 0.0760 | NDCG@10: 0.0376
- **Best Round**: 8

### Soft Domain Weighting (Seed 123)
- **Best Server Val**: MRR: 0.0225 | Recall@10: 0.0600 | NDCG@10: 0.0310
- **Final Test**: MRR: 0.0262 | Recall@10: 0.0760 | NDCG@10: 0.0376
- **Best Round**: 7

### Baseline FedAvg (Seed 256)
- **Best Server Val**: MRR: 0.0225 | Recall@10: 0.0625 | NDCG@10: 0.0316
- **Final Test**: MRR: 0.0276 | Recall@10: 0.0800 | NDCG@10: 0.0396
- **Best Round**: 8

### Soft Domain Weighting (Seed 256)
- **Best Server Val**: MRR: 0.0225 | Recall@10: 0.0625 | NDCG@10: 0.0315
- **Final Test**: MRR: 0.0276 | Recall@10: 0.0800 | NDCG@10: 0.0395
- **Best Round**: 5

---

## 5. Final Comparison Table (Final Test Metrics)

| Model | Seed | Final Test MRR | Final Test Recall@10 | Final Test NDCG@10 | Best Round |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline FedAvg** | 42 | 0.0271 | 0.0780 | 0.0387 | 6 |
| **Soft Domain** | 42 | 0.0259 | 0.0780 | 0.0378 | 8 |
| **Baseline FedAvg** | 123 | 0.0262 | 0.0760 | 0.0376 | 8 |
| **Soft Domain** | 123 | 0.0262 | 0.0760 | 0.0376 | 7 |
| **Baseline FedAvg** | 256 | 0.0276 | 0.0800 | 0.0396 | 8 |
| **Soft Domain** | 256 | 0.0276 | 0.0800 | 0.0395 | 5 |
