## **Slide 1: Title Slide**

* **Project Title:** Quality-Aware and Domain-Aware Federated Learning for Retrieval-Augmented Generation 
* **Presented by:** Abishek Chakravarthy
* **Roll No:** CS22B2054 
* **Internal Guide:** Dr. Dinesh Rajavelu 
* **Department:** Dept. of CSE, IIITDM Kancheepuram 

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

* [Insert Image: Scanned copy of the Weekly Review Progress Sheet]

---

## **Slide 4: Introduction**

* **RAG Concept**: Combines Retriever, Generator, and Knowledge Store.
* **Component Roles**: Retriever fetches facts; Generator localizes context.
* **Data Scarcity**: Fine-tuning RAG requires diverse, high-volume (query, doc) pairs.
* **Privacy Constraints**: Medical/Enterprise data cannot be shared (HIPAA/GDPR).
* **Federated Learning (FL)**: Collaborative training without raw data exchange.
* **Current Standard**: FedAvg algorithm weights updates purely by dataset size.

**[DIAGRAM DESCRIPTION: RAG + FL Architecture]**
* **Retriever/Generator Loop**: Draw an LLM box (Generator) and a smaller Encoder box (Retriever). Show an arrow from Retriever pointing to a Knowledge Store (stack of cylinders) and a "Context" arrow flowing back to the Generator.
* **Federated Nodes**: Draw 3 separate circles (Clients). Inside each, show a magnifying glass symbol (Retriever). 
* **Central Server**: Draw a Cloud/Server icon at the top. Arrows showing "Weights" moving from Clients to Server.

---

## **Slide 5: Problem Definition**

* **FedAvg Logic**: $w_j = n_j / N$ (Weights = Size ratio).
* **Issue 1 - Quality Blindness**: Noisy/corrupted data gets full influence.
* **Issue 2 - Domain Dilution**: Irrelevant domains (e.g., science data in medical RAG) pollute weights.
* **Objective**: Introduce **Quality-Awareness** using training loss signals.
* **Significance**: Improve retrieval precision without exposing private data.

**[DIAGRAM DESCRIPTION: Problem Visualization]**
* **Naive FedAvg**: Show 3 clients. Client 1 (Clean/Small), Client 2 (Clean/Medium), Client 3 (Noisy/Huge).
* **Weighting Error**: Draw the final "Global Model" arrow being heavily influenced (thickest arrow) by the "Noisy/Huge" client. Use a red "X" to show this is undesirable.

---

## **Slide 6: Literature Survey**

* **McMahan (2017)**: Introduced FedAvg; established size-based aggregation.
* **Lewis (2020)**: Defined RAG; standard for knowledge-intensive NLP.
* **Wang (2023)**: LSR Training; Retriever learns from Generator's prob-distribution.
* **Bui (2024)**: FedRAG Library; provided basic FL-RAG infra via Flower.
* **Thakur (2021)**: BEIR Benchmark; standardized IR evaluation (NFCorpus/SciFact).
* **Gaps**: Existing FL-RAG ignores client training quality and domain relevance.

---

## **Slide 7: Contributions / Work Done**

* **Infrastructure Fixes**: Resolved 4 critical bugs in FedRAG library.
* **Gradient Flow Restoration**: Fixed severed graphs and prefetch collisions.
* **QA-FedAvg Strategy**: Developed custom Flower strategy for quality-weighting.
* **Mathematical Innovation**: Formula blending inverse-loss and dataset size.
* **Baseline Framework**: Implemented Centralized, Fed-IID, and Fed-NonIID paths.

---

## **Slide 8: QA-FedAvg Methodology**

* **Blending Formula**: $w_j = \frac{\alpha \cdot q_j + (1-\alpha) \cdot (n_j / N)}{\sum [\alpha \cdot q_m + (1-\alpha) \cdot (n_m / N)]}$
* **Quality Score ($q_j$)**: Inverse-loss weight $q_j = \frac{1/ (\mathcal{L}_j + \epsilon)}{\sum 1/ (\mathcal{L}_k + \epsilon)}$
* **LSR Training Loss ($\mathcal{L}_j$)**: $D_{KL}(P_{LM} \| P_{R})$ (Generator vs Retriever)
* **Inverse-Loss Intuition**: Lower loss → Higher retriever-generator alignment → Higher weight.
* **Efficiency**: Zero extra communication; uses existing local training metrics.

**[DIAGRAM DESCRIPTION: QA-FedAvg Weighting Logic]**
* **The Scale**: Draw a mechanical balance scale. On one side, place a "Data Size" icon (stack of papers). On the other, place a "Loss Meter" (dial pointing to low). 
* **The Alpha Knob**: Draw a slider/knob labeled "α". Show it shifting weight influence toward the "Loss Meter" (Quality).

---

## **Slide 9: Analysis / Results (Baselines)**

| Method | Convergence | Loss Reduction | Key Insight |
|--------|-------------|----------------|-------------|
| **Centralized** | Rapid | 29% | Theoretical upper bound. |
| **Fed-IID** | Steady | 17% | Base Federated performance. |
| **Fed-NonIID** | Sluggish | 11% | Domain gap slows learning. |

**[DIAGRAM DESCRIPTION: Baseline Comparison Plot]**
* **Type**: Line chart.
* **X-Axis**: Rounds (1, 2, 3). **Y-Axis**: Training Loss.
* **Lines**: Centralized (steepest drop, solid line), Fed-IID (middle, dashed), Fed-NonIID (shallowest drop, dotted).

---

## **Slide 10: Analysis / Results (QA-FedAvg Sweep)**

* **Optimal Alpha**: **$\alpha = 0.6$** achieved maximum loss reduction.
* **Performance Gain**: **7.3% improvement** in loss over standard FedAvg.
* **Outlier Suppression**: Noisy client's weight reduced by **32%**.
* **Over-correction**: Pure quality ($\alpha = 1.0$) caused instability.

**[DIAGRAM DESCRIPTION: Weight Comparison Bar Chart]**
* **Contrast**: Show two bars side-by-side for the "Noisy Client". 
* **Bar 1**: FedAvg ($\alpha = 0$) → Height at 33%.
* **Bar 2**: QA-FedAvg ($\alpha = 0.6$) → Height at 22%. Label the 11% gap as "Suppressed Noise".

---

## **Slide 11: Prototype / Implementation**

* **Core Stack**: Python, Flower (flwr), HuggingFace, PyTorch.
* **Models**: all-MiniLM-L6-v2 (Retriever), DistilGPT2 (Teacher Generator).
* **Implementation**: `QualityAwareFedAvg` class (overrides `aggregate_fit`).
* **Experimental Data**: NFCorpus (training) and SciFact (heterogeneity).

**[DIAGRAM DESCRIPTION: Implementation Stack]**
* **Logo Grid**: Arrange logos for Python, PyTorch, HuggingFace, and Flower in a neat 2x2 grid.
* **Code Flow**: Brief flowchart: `Data Loader` → `Local Train` → `Loss to Server` → `QA-Aggregation`.

---

## **Slide 12: Conclusion**

* **Proven Efficacy**: QA-FedAvg reliably improves Non-IID convergence.
* **Intrinsic Signal**: Training loss is a viable proxy for client quality.
* **Robustness**: Successfully handled synthetic noise injection.
* **Correctness**: FedAvg recovered at $\alpha = 0$; baseline logic verified.

---

## **Slide 13: Future Work**

* **DAS-FedAvg**: Filter clients using cosine similarity of domain centroids.
* **Dynamic Alpha**: Adaptive $\alpha$ based on round-wise loss variance.
* **Post-Agg Evaluation**: Track MRR/NDCG after weights are merged.
* **Scaling**: Testing on larger multi-domain BEIR clusters (10+ clients).

**[DIAGRAM DESCRIPTION: DAS-FedAvg Selection]**
* **Circles in/out**: Circle for Central Server. Surround it with 5 small Client circles. 
* **Check/X**: Mark 3 close clients with a Green Check (Selected). Mark 2 distant clients with a Red X (Filtered by Domain Similarity).

---

## **Slide 14: Final Slide**

* **Thank You**
* **Any Questions?** 
