# Midsem Presentation — Slide-by-Slide Guide

**Project**: Quality-Aware and Domain-Aware Federated Retrieval-Augmented Generation  
**Time**: 7 min presentation + 3 min Q&A  
**Max slides**: 15 (including title + thank you)

---

## Slide 1: Title Slide

**Content:**
- **Title**: Quality-Aware and Domain-Aware Federated Learning for Retrieval-Augmented Generation
- **Name, Roll No**
- **Internal Guide**: `<Guide Name>`
- **Department**: Dept. of CSE, IIITDM Kancheepuram

**🗣️ What to say** (~15 sec):
> "Good morning. My project is on improving federated learning for RAG systems — specifically, making the federation smarter about which clients contribute the most useful updates."

---

## Slide 2: Table of Contents

**Content:**
- Introduction & Motivation
- Problem Definition
- Literature Survey
- System Architecture & Training Flow
- Contributions (QA-FedAvg)
- Baseline Results
- QA-FedAvg Results
- Future Work

**🗣️ What to say** (~10 sec):
> "Here's an overview of what I'll cover. I'll start with the motivation, explain the problem, show the system architecture, and then present my results comparing standard FedAvg with our quality-aware approach."

---

## Slide 3: Weekly Review Report

- Attach photo of signed weekly review report

---

## Slide 4: Introduction & Motivation

**Content (bullets):**
- RAG = Retriever + Generator + Knowledge Store
- Retriever finds relevant docs → Generator produces grounded answers
- Fine-tuning RAG requires large, diverse training data
- Data is scattered across organizations (hospitals, enterprises)
- **Federated Learning** enables collaborative training without sharing raw data
- Problem: Current FL for RAG uses **FedAvg** — weights clients only by dataset size

**📊 Diagram**: Include a simple RAG pipeline diagram:

```
┌─────────┐     ┌────────────────┐     ┌───────────┐
│  Query  │────▶│   Retriever    │────▶│ Top-k Docs│
└─────────┘     │ (Encoder Model)│     └─────┬─────┘
                └────────────────┘           │
                                             ▼
                                     ┌───────────────┐
                                     │   Generator   │
                                     │    (LLM)      │
                                     └───────┬───────┘
                                             ▼
                                     ┌───────────────┐
                                     │   Response    │
                                     └───────────────┘
```

**🗣️ What to say** (~45 sec):
> "RAG systems combine a retriever that finds relevant documents with a generator LLM that produces grounded answers. To fine-tune these models, you need large training datasets — but in practice, this data is distributed across organizations that can't share it due to privacy regulations like HIPAA or GDPR. Federated Learning solves this by keeping data local and only sharing model updates. However, the standard aggregation method, FedAvg, treats all clients equally based on dataset size — which is a problem when some clients have noisy or irrelevant data."

---

## Slide 5: Problem Definition

**Content (bullets):**
- **FedAvg aggregation**: $w_j = n_j / \sum n_k$ (dataset size only)
- **Problem 1**: A noisy client with many examples gets disproportionate influence
- **Problem 2**: Clients with irrelevant domain data dilute the global model
- **Our goal**: Replace naive weighting with **quality-aware** aggregation
- Use training loss as a quality signal already available during FL

**📊 Diagram**: Side-by-side comparison showing:

```
Standard FedAvg:                    Quality-Aware FedAvg:
┌──────────┐  w=0.33               ┌──────────┐  w=0.39
│ Client 0 │──────┐                │ Client 0 │──────┐
│ (Clean)  │      │                │ (Clean)  │      │
└──────────┘      ▼                └──────────┘      ▼
┌──────────┐  ┌────────┐          ┌──────────┐  ┌────────┐
│ Client 1 │─▶│ Global │          │ Client 1 │─▶│ Global │
│ (Clean)  │  │ Model  │          │ (Clean)  │  │ Model  │
└──────────┘  └────────┘          └──────────┘  └────────┘
┌──────────┐      ▲                ┌──────────┐      ▲
│ Client 2 │──────┘  w=0.33       │ Client 2 │──────┘  w=0.23
│ (Noisy!) │                      │ (Noisy!) │  ⬇️ down-weighted!
└──────────┘                      └──────────┘
```

**🗣️ What to say** (~45 sec):
> "The core problem is that FedAvg weights every client proportional to their dataset size. If Client 2 has noisy or mismatched data, it still gets equal weight — one-third of the influence. This drags down the global model. Our solution is quality-aware aggregation: we use the training loss, which is already computed during federated learning, as a quality signal. Clients with lower loss — meaning better local training — get higher weight. Noisy clients are automatically down-weighted."

---

## Slide 6: Literature Survey

**Content (bullets):**
- **FedAvg** (McMahan et al., 2017) — foundational FL algorithm, size-weighted aggregation
- **FedProx** (Li et al., 2020) — adds proximal term for heterogeneous data
- **FedRAG** (Bui et al., 2024) — first FL framework for RAG systems using LSR/RALT training
- **LSR Training** (Wang et al.) — retriever learns from generator's utility signal via KL divergence
- **Gap**: No existing work combines quality-aware aggregation with federated RAG training

**🗣️ What to say** (~30 sec):
> "FedAvg is the standard algorithm. FedProx handles data heterogeneity but doesn't use quality signals. The FedRAG library provides FL infrastructure for RAG but uses standard FedAvg internally. Our contribution fills the gap: quality-aware aggregation specifically designed for federated RAG."

---

## Slide 7: System Architecture & LSR Training

**Content (bullets):**
- Built on **FedRAG** library + **Flower** FL framework
- **LSR Training**: Retriever learns what helps the generator
  - Generator scores: $P(\text{response} \mid \text{query, doc})$ → teacher signal
  - Retriever scores: $\cos(q_{\text{emb}}, d_{\text{emb}})$ → student
  - Loss: $\text{KL}(\text{softmax}(\text{retriever}), \text{softmax}(\text{generator}))$
- **Fed round**: Broadcast → Local train → Collect (weights, loss) → Aggregate

**📊 Diagram**: Federated training round flow:

```
    ┌────────────────────────────────────────────┐
    │              FL SERVER                      │
    │  Global Model θ + QA-FedAvg Strategy       │
    └──────┬──────────────┬──────────────┬───────┘
           │ broadcast θ  │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Client 0 │   │ Client 1 │   │ Client 2 │
    │ Local    │   │ Local    │   │ Local    │
    │ LSR Train│   │ LSR Train│   │ LSR Train│
    │ loss=0.08│   │ loss=0.08│   │ loss=0.15│
    └──────┬───┘   └────┬─────┘   └────┬─────┘
           │ (θ₀,L₀)    │ (θ₁,L₁)      │ (θ₂,L₂)
           └──────┬──────┴──────────────┘
                  ▼
    ┌────────────────────────────────────────────┐
    │  QA-FedAvg: w_j = α·q_j + (1-α)·(n_j/N) │
    │  q_j = (1/L_j) / Σ(1/L_k)                │
    │  → noisy Client 2 gets LOWER weight       │
    └────────────────────────────────────────────┘
```

**🗣️ What to say** (~50 sec):
> "The system uses the FedRAG library built on Flower. LSR training works by treating the generator as a teacher: it scores how useful each retrieved document is, and the retriever learns to match that ranking via KL divergence. In the federated setup, each client trains locally and reports back its updated weights along with its training loss. Our QA-FedAvg strategy uses this loss as a quality signal — clients with lower loss, meaning better training quality, receive higher aggregation weight. The parameter alpha controls the balance: at alpha=0 we recover standard FedAvg, at alpha=1 we use pure quality weighting."

---

## Slide 8: Our Contribution — QA-FedAvg

**Content (bullets):**
- **QA-FedAvg formula:**
  $$w_j = \frac{\alpha \cdot q_j + (1-\alpha) \cdot \frac{n_j}{N}}{\sum_m [\alpha \cdot q_m + (1-\alpha) \cdot \frac{n_m}{N}]}$$
  where $q_j = \frac{1/L_j}{\sum_k 1/L_k}$ (inverse-loss quality score)
- **α = 0** → standard FedAvg (baseline recovery)
- **α = 0.6** → optimal balance (validated experimentally)
- **α = 1** → pure quality (over-correction risk)
- Implemented as custom Flower `Strategy` — ~80 lines of code
- Drops in as a replacement for [FedAvg](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/venv/lib/python3.12/site-packages/flwr/server/strategy/fedavg.py#52-285) — no changes to client code

**🗣️ What to say** (~40 sec):
> "Our key contribution is this formula. The quality score q_j is the normalized inverse loss — lower loss means higher quality. Alpha controls the trade-off. At alpha=0, we recover FedAvg exactly. At alpha=1, it's pure quality weighting. We found alpha=0.6 gives the best balance. The implementation is a custom Flower Strategy class that overrides the aggregation step — it's a drop-in replacement requiring zero changes to the client code."

---

## Slide 9: Baseline Results

**Content (table):**

| Method | R1 Loss | R2 Loss | R3 Loss | MRR | Recall@10 |
|--------|---------|---------|---------|-----|-----------|
| Centralized | 0.001037 | 0.000873 | 0.000739 | 0.0664 | 0.1325 |
| Fed-IID | 0.001066 | 0.000971 | 0.000882 | 0.0643 | 0.1205 |
| Fed-NonIID | 0.001008 | 0.000951 | 0.000893 | 0.0643 | 0.1205 |

**Key observations (bullets):**
- Centralized converges fastest (29% loss reduction)
- Fed-IID: 17% loss reduction, metrics stagnant in 3 rounds
- Fed-NonIID: slowest (11%), cross-domain client dilutes updates
- **The "FL Gap"**: Federated < Centralized → room for improvement

**📊 Diagram**: Bar chart or line plot of loss convergence across 3 rounds for all 3 methods.

**🗣️ What to say** (~40 sec):
> "These are our baseline results using NFCorpus medical data with 3 clients over 3 rounds. Centralized training converges the fastest with a 29% loss reduction and improving retrieval metrics. Federated IID is slower at 17%, and federated non-IID — where Client 2 has science data instead of medical — is the slowest at only 11%. This gap between federated and centralized is exactly what QA-FedAvg targets."

---

## Slide 10: QA-FedAvg Results — Alpha Sweep

**Content (table):**

| α | R1 Loss | R2 Loss | R3 Loss | Δ vs FedAvg |
|---|---------|---------|---------|-------------|
| **0.0 (FedAvg)** | 0.001183 | 0.001068 | **0.000973** | baseline |
| 0.2 | 0.001167 | 0.001044 | 0.000943 | -3.1% |
| 0.4 | 0.001153 | 0.001022 | 0.000918 | -5.7% |
| **0.6** | **0.001144** | **0.001005** | **0.000902** | **-7.3%** 🏆 |
| 1.0 | 0.001157 | 0.001033 | 0.000932 | -4.2% |

**Key observations (bullets):**
- **α=0.6 achieves the best loss**: 7.3% improvement over FedAvg
- α=1.0 shows **over-correction** — too aggressive quality weighting
- Classic **inverted-U** pattern: moderate α is optimal
- α=0.6 R3 loss (0.000902) approaches centralized (0.000739)

**📊 Diagram**: Line plot showing R3 loss vs alpha — the inverted-U curve with minimum at α=0.6.

**🗣️ What to say** (~45 sec):
> "Here are the alpha sweep results with a noisy client in the federation. At alpha=0, standard FedAvg gives all clients equal weight, leading to a final loss of 0.000973. As we increase alpha, quality-aware weighting kicks in and loss decreases — the optimal point is alpha=0.6 with a 7.3% improvement. But at alpha=1.0, we see over-correction: pure quality weighting is too aggressive and performance actually degrades. This inverted-U pattern is common in regularization research — a moderate balance outperforms both extremes."

---

## Slide 11: QA-FedAvg — Weight Redistribution

**Content (table):**

| α | Client 0 (Clean) | Client 1 (Clean) | Client 2 (Noisy) |
|---|:-:|:-:|:-:|
| 0.0 | 0.332 | 0.332 | **0.336** |
| 0.2 | 0.355 | 0.350 | **0.295** |
| 0.4 | 0.373 | 0.366 | **0.261** |
| 0.6 | 0.390 | 0.382 | **0.228** |
| 1.0 | 0.415 | 0.405 | **0.180** |

**Key observations (bullets):**
- At α=0: noisy client gets **equal weight** (0.336) — blindly trusted
- At α=0.6 (optimal): noisy client down-weighted to **0.228** (32% reduction)
- At α=1.0: noisy client weight drops to **0.180** — but over-correction hurts
- QA-FedAvg **automatically detects and penalizes** low-quality clients

**📊 Diagram**: Stacked bar chart showing weight distribution across alphas, with Client 2's bar shrinking as alpha increases.

**🗣️ What to say** (~30 sec):
> "This table shows how the aggregation weights shift. At alpha=0, the noisy client gets one-third of the influence. As alpha increases, QA-FedAvg automatically detects that Client 2 has higher loss and reduces its weight. At the optimal alpha=0.6, the noisy client's influence drops by 32%. The quality signal — just the training loss — is enough to identify and penalize bad clients."

---

## Slide 12: Prototype / Implementation

**Content (bullets):**
- Built on open-source: **FedRAG** library + **Flower** FL framework
- **Key files implemented:**
  - [quality_aware_fedavg.py](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/quality_aware_fedavg.py) — Custom [QualityAwareFedAvg](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/quality_aware_fedavg.py#28-125) strategy (~80 lines)
  - [federated_iid_qa.py](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_iid_qa.py) — Federated training runner with alpha config
  - [prepare_beir_data.py](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/baseline/prepare_beir_data.py) — BEIR dataset loading, knowledge store, evaluation
- **Bug fixes to FedRAG** (3 gradient flow bugs in retriever training)
- Dataset: **NFCorpus** (Medical/Nutrition domain) from BEIR benchmark
- Model: **all-MiniLM-L6-v2** (retriever) + **distilgpt2** (generator)

**🗣️ What to say** (~25 sec):
> "The implementation uses the FedRAG library and Flower framework. I implemented the QA-FedAvg strategy as a custom Flower Strategy class and fixed three critical gradient flow bugs in the FedRAG library that were preventing the retriever from actually learning. The experiments use the NFCorpus medical dataset from the BEIR benchmark."

---

## Slide 13: Conclusion & Future Work

**Content (bullets):**

**Conclusions:**
- ✅ QA-FedAvg validated: α=0.6 achieves **7.3% lower loss** than FedAvg
- ✅ Automatic detection and down-weighting of noisy clients
- ✅ α=0.0 perfectly recovers standard FedAvg (correctness verified)
- ✅ Framework is a drop-in replacement — no client code changes needed

**Future Work (Next 2 months):**
- **Domain-Aware Selection (DAS-FedAvg)**: Selective client participation based on data domain similarity — exclude irrelevant domain clients entirely
- **Post-training evaluation**: MRR, Recall@k, NDCG@k evaluation after aggregation
- **Larger scale experiments**: More clients, more rounds, multiple datasets
- **Dynamic α**: Auto-tune α based on observed client heterogeneity per round

**🗣️ What to say** (~30 sec):
> "In conclusion, QA-FedAvg successfully improves federated RAG training by using training loss as a quality signal for aggregation. The optimal alpha=0.6 gives a 7.3% improvement and automatically handles noisy clients. For the next two months, I plan to implement domain-aware client selection — where clients from irrelevant domains are excluded entirely — and run larger-scale experiments with post-training retrieval evaluation."

---

## Slide 14: References

**Content:**
- McMahan, B. et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
- Li, T. et al. "Federated Optimization in Heterogeneous Networks." MLSys 2020.
- Bui, N. "FedRAG: Federated Retrieval-Augmented Generation." GitHub, 2024.
- Wang, Z. et al. "Learning to Retrieve for Retrieval-Augmented Generation." arXiv, 2023.
- Reimers, N. and Gurevych, I. "Sentence-BERT." EMNLP 2019.

---

## Slide 15: Thank You

- Thank You
- Any Questions?

---

## 🎯 Timing Guide

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 15 sec |
| 2 | TOC | 10 sec |
| 3 | Weekly Report | skip in talk |
| 4 | Introduction | 45 sec |
| 5 | Problem | 45 sec |
| 6 | Literature | 30 sec |
| 7 | Architecture | 50 sec |
| 8 | QA-FedAvg Formula | 40 sec |
| 9 | Baseline Results | 40 sec |
| 10 | Alpha Sweep Results | 45 sec |
| 11 | Weight Redistribution | 30 sec |
| 12 | Implementation | 25 sec |
| 13 | Conclusion + Future | 30 sec |
| **Total** | | **~6.5 min** ✅ |

---

## 🛡️ Likely Viva Questions & Answers

**Q: Why are the metric magnitudes so low?**
> "The small magnitudes reflect our constrained setup — 500 training pairs, 1000 docs, 3 rounds — designed for rapid iteration. The LSR loss is inherently small because it's KL divergence between softmaxed distributions. What matters is the relative trends, which match theoretical expectations."

**Q: Why alpha=0.6 and not some other value?**
> "We swept alpha from 0 to 1 in increments of 0.2. Alpha=0.6 gave the minimum loss — it's the optimal balance between quality awareness and dataset-size weighting. Too little alpha and you ignore quality; too much and you over-correct based on noisy single-round loss estimates."

**Q: How does this compare to FedProx?**
> "FedProx adds a proximal regularization term to handle data heterogeneity — it constrains how far local models drift from the global model. Our approach is complementary: we change the aggregation weights, not the local training objective. They could be combined."

**Q: What's the overhead of quality-aware aggregation?**
> "Near zero. The only additional computation is computing inverse-loss quality scores during aggregation — about 5 floating point operations per client per round. The loss is already reported by each client."

**Q: What does 'noisy client' mean in your experiment?**
> "Client 2 has corrupted query-document pairs — 70% of its training pairs have the response shuffled, so queries point to wrong documents. This simulates a real-world scenario where a participating organization has poorly curated or mislabeled training data."
