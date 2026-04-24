# QA-FedAvg Mechanism Benchmark — Experiment Log

> **Purpose**: Prevent going in circles. Every run batch gets one row. Check here BEFORE trying a new config.

---

## Runs (newest first)

| # | Date | Retriever | LR | Rounds | Epochs | Clients | Train | Docs | Split | Noise Levels | β | Quality Signal | Best α MRR | FedAvg MRR | Pre-train MRR | Verdict |
|---|------|-----------|-----|--------|--------|---------|-------|------|-------|-------------|---|----------------|------------|------------|---------------|---------|
| **1** | Apr 2026 | `paraphrase-MiniLM-L3-v2` | 2e-6 | 4 | 1 | 5 | 4000 | 8000 | equal | 0,0,0,0.3,0.7 | 5.0 | rank-displacement | 0.0234 (α=1.0) | 0.0226 (α=0.0) | **0.0237** | ❌ Training degrades retriever (final < pre-train). Best round=1 for α=0.0,0.3. Quality signal doesn't consistently discriminate noisy clients (acceptance test `lower_quality_clients_downweighted` fails for all α>0). |

---

## What Has NOT Been Tried (within the mechanism benchmark redesign)

- LR below 2e-6 (e.g., 1e-6, 5e-7)
- Stronger retriever (`all-MiniLM-L6-v2`) with the quality ladder noise
- Stronger noise levels (e.g., 0/0/0/0.5/0.9)
- Fewer training pairs (e.g., 1000–2000)
- Fewer rounds (e.g., 2)
- Using holdout evaluation loss as the quality signal instead of rank-displacement
- Non-federated diagnostic to verify LSR training can improve retrieval at all
