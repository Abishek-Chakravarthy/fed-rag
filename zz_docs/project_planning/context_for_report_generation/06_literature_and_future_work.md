# 06 — Literature Survey, Conclusion & Future Work

## 1. Literature Survey

### 1.1 Federated Learning
- **McMahan, B. et al. (2017)**. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*. — Introduced FedAvg, the foundational FL algorithm where clients train locally and the server aggregates model updates weighted by dataset size.
- **Li, T. et al. (2020)**. "Federated Optimization in Heterogeneous Networks." *MLSys 2020*. — Proposed FedProx, which adds a proximal regularization term to handle data and systems heterogeneity, constraining local models from drifting too far from the global model.
- **Zhao, Y. et al. (2018)**. "Federated Learning with Non-IID Data." *arXiv:1806.00582*. — Demonstrated that non-IID data distribution across clients significantly degrades FedAvg performance, motivating solutions like data sharing or weighted strategies.

### 1.2 Retrieval-Augmented Generation
- **Lewis, P. et al. (2020)**. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. — Introduced RAG: combining a pre-trained seq2seq model with a retriever to generate factual, grounded responses.
- **Wang, Z. et al. (2023)**. "Learning to Retrieve for Retrieval-Augmented Generation." *arXiv*. — LM-Supervised Retrieval (LSR): the retriever learns to rank documents based on the generator's utility signal using KL divergence.

### 1.3 Federated RAG
- **Bui, N. (2024)**. "FedRAG: Federated Retrieval-Augmented Generation." *GitHub*. https://github.com/nerdai/fed-rag — Open-source library providing FL infrastructure for RAG systems with LSR and RALT training, built on Flower. Our project builds directly on this library.

### 1.4 Sentence Transformers
- **Reimers, N. and Gurevych, I. (2019)**. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*. — Efficient framework for computing sentence embeddings, used as our retriever backbone (all-MiniLM-L6-v2).

### 1.5 BEIR Benchmark
- **Thakur, N. et al. (2021)**. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS 2021*. — Standardized IR benchmark with 18 datasets. We use NFCorpus (medical) and SciFact (scientific claims).

### 1.6 Research Gap
No existing work combines **quality-aware aggregation** with **federated RAG training**. FedProx addresses heterogeneity through regularization (local training objective change) but does not use quality signals during aggregation. Our QA-FedAvg is complementary — it changes the aggregation weights, not the local training objective.

---

## 2. Conclusion (Mid-Semester)

1. **Framework Validated**: We successfully implemented QA-FedAvg as a drop-in Flower Strategy replacement. At α=0.0, it exactly reproduces standard FedAvg, confirming correctness.
2. **Quality Detection Works**: QA-FedAvg automatically identifies low-quality (high-loss) clients and reduces their aggregation influence — at α=0.6, the noisy client's weight drops by 32%.
3. **7.3% Improvement**: α=0.6 achieves the best global loss (0.000902 vs. 0.000973 for FedAvg), a 7.3% reduction.
4. **Over-Correction Risk Identified**: α=1.0 (pure quality weighting) actually performs worse than α=0.6, demonstrating the need for balanced weighting.
5. **Engineering Contributions**: Fixed 4 critical bugs in the FedRAG library that completely prevented retriever training, enabling the first working federated LSR training pipeline.

---

## 3. Future Work (Next Two Months)

### 3.1 Domain-Aware Client Selection (DAS-FedAvg)
**Goal:** Exclude clients whose data domain is irrelevant to the target training objective.

**Approach:**
1. **Domain Profiling**: Compute a domain centroid for each client by averaging the embeddings of their knowledge store documents: $\bar{\mathbf{e}}_j = \frac{1}{|D_j|}\sum_{d \in D_j} \text{encode}(d)$
2. **Domain Similarity**: Compute cosine similarity between client centroids and the target domain centroid: $d_j = \cos(\bar{\mathbf{e}}_j, \bar{\mathbf{e}}^{\text{target}})$
3. **Selective Participation**: Only clients with $d_j > \tau$ (threshold) participate in each round.
4. **Combined Aggregation**: Apply QA-FedAvg weighting over the selected subset:
   $$\theta^{t+1} = \sum_{j \in S^t} w_j \cdot \theta_j^t, \quad S^t = \{j \mid d_j > \tau\}$$

### 3.2 Post-Training Retrieval Evaluation
Currently we only evaluate retrieval metrics (MRR, Recall@k, NDCG@k) before training. We plan to evaluate after each federated round to show that QA-FedAvg not only reduces loss but also improves actual retrieval quality.

### 3.3 Larger-Scale Experiments
- Increase from 3 to 10+ clients
- Increase from 3 to 10+ federated rounds
- Test on multiple BEIR datasets simultaneously
- Measure communication efficiency (fewer rounds to target performance)

### 3.4 Dynamic α Tuning
Auto-tune α per round based on observed client loss variance. High variance (heterogeneous quality) → increase α. Low variance (homogeneous quality) → decrease α toward FedAvg.

---

## 4. Comparison Summary Table (For Report)

| Method | Aggregation Weight $w_j$ | Client Selection | Loss Reduction | Status |
|--------|--------------------------|-----------------|----------------|--------|
| Centralized (upper bound) | N/A (all data pooled) | N/A | 29% | ✅ Done |
| FedAvg (IID baseline) | $n_j / N$ | All clients | 17% | ✅ Done |
| FedAvg (Non-IID baseline) | $n_j / N$ | All clients | 11% | ✅ Done |
| **QA-FedAvg (ours)** | $\alpha \cdot q_j + (1-\alpha) \cdot n_j/N$ | All clients | **7.3% better than FedAvg** | ✅ Done |
| DAS-FedAvg (ours, future) | QA-FedAvg over selected | $d_j > \tau$ only | TBD | 🔲 Planned |
