# DAS-FedAvg Experiment Log

## Overview
This log documents the findings from the initial run of the **Domain-Aware Selective FedAvg (DAS-FedAvg)** mechanism. The experiments were run to evaluate whether domain-aware client filtering based on a threshold (`tau`) can prevent domain dilution when aggregating weights in a federated RAG setting.

## Experiment Configuration
- **Target Domain**: NFCorpus (Medical)
- **Clients**: 5 total 
  - Client 0: Medical (NFCorpus)
  - Client 1: Science (SciFact)
  - Client 2: Finance (FiQA)
  - Client 3: Argument (ArguAna)
  - Client 4: Biomedical
- **Rounds**: 4
- **Evaluated Tau Values**: 0.00, 0.20, 0.40, and 0.60

## Experiment Results Data

### Client Domain & Relevance Scores
| Client ID | Domain | Number of Training Pairs | Domain Relevance Score |
| :--- | :--- | :--- | :--- |
| **0** | Medical | 4000 | `1.0000` |
| **1** | Science | 237 | `0.4791` |
| **4** | Biomedical | 100 | `0.3212` |
| **3** | Argument | 980 | `0.0422` |
| **2** | Finance | 166 | `0.0130` |

### Evaluated Tau Thresholds & Client Selection
| Tau ($\tau$) | Selected Clients (Round 2+) | Excluded Domains |
| :--- | :--- | :--- |
| **0.00** | 0, 1, 2, 3, 4 | *None (All Participate)* |
| **0.20** | 0, 1, 4 | Finance, Argument |
| **0.40** | 0, 1 | Finance, Argument, Biomedical |
| **0.60** | 0 | Finance, Argument, Biomedical, Science |

### Federated Evaluation Metrics (Target = Medical)
> [!NOTE]
> Evaluation metrics remained identical across all $\tau$ experiments. This is because Round 1 acts as a "fallback" phase where all proxy clients are evaluated. As model performance steadily decayed in later rounds, the system always saved Round 1 as the "best model," completely masking the effects of selective $\tau$ filtering in the final evaluation.

| Metric | Pre-training Baseline | Best Server-Val (Round 1) | Final Test |
| :--- | :--- | :--- | :--- |
| **MRR** | 0.0159 | 0.0112 | 0.0117 |
| **NDCG@10** | 0.0209 | 0.0162 | 0.0173 |
| **Recall@10** | 0.0375 | 0.0325 | 0.0360 |

## What Went Right
1. **Accurate Domain Relevance Computation**: The pre-computed cosine similarity between client centroids and the target centroid successfully mapped to semantic expectations:
   - Medical: `1.000`
   - Science: `0.479`
   - Biomedical: `0.321`
   - Argument: `0.042`
   - Finance: `0.013`
2. **Selective Participation (DAS-FedAvg Logic)**: The Flower strategy accurately enforced the `tau` threshold. For `tau=0.60`, from Round 2 onwards, only Client 0 (Medical) was selected for training, cleanly excluding the out-of-domain clients.
3. **End-to-End Pipeline Health**: The code seamlessly integrated multi-domain dataset preparation, distributed proxy logic, and post-aggregation evaluation on the target domain.

## What is Yet to be Achieved
1. **Model Performance Degradation**: Across all rounds, the retriever's performance decreased relative to the pre-trained weights (Pre-train Target MRR: ~0.0159 → Round 1 MRR: ~0.0112 → Round 2+ MRR: ~0.000 - 0.001). We need to tune hyperparameters (e.g., learning rate, weight decay, freezing lower layers) to prevent catastrophic forgetting.
2. **"Best Model" Edge Case with Round 1**: Due to Flower's proxy client architecture, `DAS-FedAvg` currently maps `proxy_cid` to `logical_cid` during the first `aggregate_fit`. As a result, **all clients are selected in Round 1 regardless of `tau`**. Since model performance degraded in subsequent rounds, the system correctly identified Round 1 as the "best round" for both `tau=0.00` and `tau=0.60`. This resulted in identical final test metrics across both sweeps, masking any benefits of DAS filtering.
3. **Pre-registering Logical CIDs**: To enforce `tau` from the very first round, we should look into passing logical CIDs during client initialization or pre-registering them, so Round 1 does not act as a fallback "all-client" round. 

## Next Steps
- Address the catastrophic forgetting in the retriever (adjust `learning_rate` from `2e-6`, increase `num_train_epochs`, etc.).
- Fix the Round 1 proxy-logical mapping limitation so DAS filtering kicks in immediately.
- Re-run the `tau` sweep once learning is stable to generate the target curves.
