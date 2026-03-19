## **Slide 1: Title Slide**

* 
**Project Title:** Quality-Aware and Domain-Aware Federated Learning for Retrieval-Augmented Generation 


* 
**Presented by:** `Abishek Chakravarthy` 


* 
**Roll No:** `CS22B2054` 


* 
**Internal Guide:** `Dr. Dinesh Rajavelu` 


* 
**Department:** Dept. of CSE, IIITDM Kancheepuram 



---

## **Slide 2: Table of Contents**

* Introduction 

* Problem Definition 

* Literature Survey 

* Contribution 

* Progress / Work Done 

* Results / Analysis 

* Prototype / Products 

* Conclusion 

* Future Work 



---

## **Slide 3: Weekly Review Report**

* Attach Picture of Weekly review report 



---

## **Slide 5: Introduction**

* RAG (Retrieval-Augmented Generation) = Retriever + Generator + Knowledge Store
* Retriever finds relevant documents → Generator produces grounded, factual answers
* Fine-tuning RAG requires large, diverse (query, document) training pairs
* Data is scattered across organizations (hospitals, enterprises) — cannot be shared (HIPAA, GDPR)
* **Federated Learning** enables collaborative training without sharing raw data
* Each client trains locally, shares only model weight updates with a central server
* Standard FL uses **FedAvg** — aggregates updates weighted only by dataset size



---

## **Slide 6: Problem Definition**

* **FedAvg aggregation:** $w_j = n_j / \sum n_k$ — weights clients by dataset size only
* **Problem 1 — Quality Blindness:** A noisy client with many examples gets disproportionate influence, dragging down the global model
* **Problem 2 — Domain Irrelevance:** Clients with irrelevant domain data (e.g., science data in a medical federation) dilute the global model update
* **Our Goal:** Replace naive size-only weighting with **quality-aware aggregation** that uses the training loss — a signal already available during FL — to penalize low-quality clients

**Diagram suggestion:** Side-by-side comparison showing FedAvg (equal weights) vs QA-FedAvg (noisy client down-weighted)



---

## **Slide 7: Literature Survey**

* **FedAvg** (McMahan et al., 2017) — Foundational FL algorithm, size-weighted aggregation
* **FedProx** (Li et al., 2020) — Adds proximal regularization for heterogeneous data, but does not use quality signals during aggregation
* **RAG** (Lewis et al., 2020) — Retriever + Generator architecture for knowledge-intensive tasks
* **LSR Training** (Wang et al., 2023) — Retriever learns from generator's utility signal via KL divergence
* **FedRAG** (Bui, 2024) — Open-source FL framework for RAG, uses standard FedAvg internally
* **BEIR** (Thakur et al., 2021) — Standard IR benchmark (NFCorpus, SciFact datasets used)
* **Gap:** No existing work combines quality-aware aggregation with federated RAG training



---

## **Slide 8: Contributions / Work Done**

* **Contribution 1: FedRAG Bug Fixes** — Fixed 4 critical bugs in the open-source FedRAG library that completely prevented retriever training:
  - Gradient graph severed in data collator (moved forward pass to compute_loss)
  - DataLoader prefetch invalidating computation graphs
  - Optimizer initialized with zero parameters (LSRLoss has no trainable weights)
  - Training loss discarded before reaching server (returned 0 instead of real loss)

* **Contribution 2: QA-FedAvg Strategy** — Designed and implemented quality-aware aggregation:
  - Quality score: $q_j = (1/L_j) / \sum(1/L_k)$ (inverse-loss, normalized)
  - Combined weight: $w_j = \alpha \cdot q_j + (1-\alpha) \cdot n_j/N$
  - α=0 recovers standard FedAvg; α=0.6 found optimal
  - Implemented as ~80-line custom Flower Strategy class (drop-in replacement)

* **Contribution 3: Baseline Framework** — Centralized, Fed-IID, Fed-NonIID comparisons



---

## **Slide 9: Analysis / Results**

**Baseline Comparison:**

| Method | R1 Loss | R3 Loss | Loss Reduction |
|--------|---------|---------|----------------|
| Centralized | 0.001037 | 0.000739 | 29% |
| Fed-IID | 0.001066 | 0.000882 | 17% |
| Fed-NonIID | 0.001008 | 0.000893 | 11% |

**QA-FedAvg Alpha Sweep (with noisy client):**

| α | R3 Loss | Δ vs FedAvg | Noisy Client Weight |
|---|---------|-------------|---------------------|
| 0.0 (FedAvg) | 0.000973 | — | 0.336 |
| 0.2 | 0.000943 | −3.1% | 0.295 |
| 0.4 | 0.000918 | −5.7% | 0.261 |
| **0.6 (Best)** | **0.000902** | **−7.3%** | **0.228** |
| 1.0 | 0.000932 | −4.2% | 0.180 |

**Key findings:** α=0.6 achieves best loss. Noisy client weight reduced by 32%. Over-correction at α=1.0 (inverted-U pattern).

**Diagram suggestion:** Line plot of loss convergence across rounds for each α value



---

## **Slide 10: Prototype / Products**

* Built on open-source: **FedRAG** library + **Flower** FL framework
* **Key files implemented:**
  - `quality_aware_fedavg.py` — Custom QualityAwareFedAvg Flower Strategy
  - `federated_iid_qa.py` — Federated training runner with alpha configuration
  - `prepare_beir_data.py` — BEIR dataset loading, knowledge store creation, evaluation
* **Models:** all-MiniLM-L6-v2 (retriever) + distilgpt2 (generator)
* **Dataset:** NFCorpus (Medical/Nutrition) and SciFact (Science) from BEIR benchmark
* **Evaluation:** MRR, Recall@10, NDCG@10 on held-out queries



---

## **Slide 11: Conclusion**

* ✅ QA-FedAvg validated: α=0.6 achieves **7.3% lower loss** than standard FedAvg
* ✅ Automatic detection and down-weighting of noisy clients (32% influence reduction)
* ✅ α=0.0 perfectly recovers standard FedAvg — correctness verified
* ✅ Drop-in replacement — no changes to client-side code required
* ✅ Fixed 4 critical gradient flow bugs in the FedRAG library
* ✅ Established baseline framework: Centralized > Fed-IID > Fed-NonIID convergence confirmed



---

## **Slide 12: Future Work**

* **Domain-Aware Client Selection (DAS-FedAvg):** Filter clients per round based on cosine similarity between client data domain embeddings and target domain — exclude irrelevant clients entirely
* **Post-Training Retrieval Evaluation:** Evaluate MRR, Recall@k, NDCG@k after each aggregation round (not just before)
* **Larger-Scale Experiments:** Scale to 10+ clients, 10+ rounds, multiple BEIR datasets
* **Dynamic α Tuning:** Auto-tune α per round based on observed client loss variance



---

## **Slide 13: References**

* McMahan, B. et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS, 2017.
* Li, T. et al. "Federated Optimization in Heterogeneous Networks." MLSys, 2020.
* Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS, 2020.
* Wang, Z. et al. "Learning to Retrieve for Retrieval-Augmented Generation." arXiv, 2023.
* Bui, N. "FedRAG: Federated Retrieval-Augmented Generation." GitHub, 2024.
* Thakur, N. et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models." NeurIPS, 2021.
* Reimers, N. and Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP, 2019.



---

## **Slide 14: Final Slide**

* Thank You 

* Any Questions? 
