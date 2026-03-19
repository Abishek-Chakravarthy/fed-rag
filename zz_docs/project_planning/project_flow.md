# Quality-Aware and Domain-Aware Federated RAG

---

## 1. Problem Statement

Retrieval-Augmented Generation (RAG) systems combine a **retriever** (which finds relevant documents from a knowledge store) with a **generator** (an LLM that produces answers grounded in those documents). Fine-tuning these components — using methods like LSR (retriever learns what helps the generator) and RALT (generator learns to use retrieved context) — significantly improves RAG quality.

However, fine-tuning requires **large, diverse training data**, which in practice is scattered across organizations that **cannot share it** due to privacy regulations (HIPAA, GDPR) or competitive concerns. Federated Learning (FL) solves this by keeping data local: each organization trains on its own data, and only **model weight updates** (not data) are shared.

**The core problem statement:**

> Current federated RAG systems aggregate client model updates using **FedAvg**, which weights each client's contribution **solely by dataset size**. This is naive — a client with 10,000 noisy, off-topic training examples gets 10× the influence of a client with 1,000 high-quality, domain-relevant examples. In RAG specifically, "quality" has a clear measurable meaning: **how well the retriever's selected documents help the generator produce correct answers**. This quality signal exists during training (as the LSR loss) but is currently ignored during aggregation.

**This creates two concrete problems:**

1. **Low-quality clients degrade the global model.** A hospital with poorly curated medical records contributes updates that push the retriever away from useful document rankings, and FedAvg gives those updates proportional weight regardless.

2. **Irrelevant clients waste computation and harm convergence.** In a federation spanning multiple domains (e.g., cardiology, oncology, radiology), training a cardiology-focused RAG retriever benefits from cardiology clients but is diluted or harmed by radiology clients. Standard FedAvg has no mechanism to exclude or down-weight irrelevant participants.

---

## 2. Proposed Contributions

### Core Contribution: Quality-Aware Federated Aggregation for RAG Systems

Replace FedAvg's dataset-size-only weighting with **quality-weighted aggregation**, where each client's contribution is scaled by how well its local training actually improved retrieval quality.

### Extended Contribution: Domain-Aware Federated RAG with Selective Client Participation

Build on quality-aware aggregation by introducing **selective client participation**: only clients whose data domain aligns with the current training objective participate in each round.

---

## 3. The FedRAG System — Complete Federated Flow

This section explains the end-to-end federated training flow as it currently works, built on top of the [FedRAG library](https://github.com/nerdai/fed-rag).

### 3.1 System Architecture

```
RAGSystem
├── retriever     → SentenceTransformer encoder (embeds queries + documents)
├── generator     → LLM (produces answers given query + retrieved context)
├── knowledge_store → Vector store of pre-embedded document chunks
└── rag_config    → top_k, context_separator, etc.
```

Federation wraps around centralized training:

```
RAGSystem          ← the shared model being improved
    ↓
Trainer            ← LSR (retriever) or RALT (generator)
    ↓
TrainerManager     ← orchestrates training, freezes the non-active model
    ↓
FLTask             ← wraps trainer into Flower server + client
    ↓
Flower (flwr)      ← handles round coordination, weight broadcast, aggregation
```

### 3.2 How LSR Retriever Training Works

LSR (Learned Sparse Retrieval) is the training method that teaches the retriever to rank documents by **how useful they are to the generator**.

**The training loop has two stages per batch:**

**Stage A — Data Collator (inside DataLoader, non-differentiable):**

```
For each (query, response) in the batch:
    1. Retrieve top_k documents using current retriever (non-differentiable)
       → context_texts: list[str]

    2. Score each document with the frozen generator:
       lm_score = P(response | query, document)
       → lm_scores: tensor[top_k]  (teacher signal)

    3. Return raw data: {queries, context_texts, lm_scores}
```

**Stage B — `compute_loss()` (inside training step, fully differentiable):**

```
For each query:
    1. query_embedding = model.forward(query)          ← DIFFERENTIABLE
    2. context_embedding = model.encode(context_texts)  ← no_grad
    3. retrieval_scores = cosine_sim(query_emb, ctx_emb) ← DIFFERENTIABLE

loss = KL_divergence(softmax(retrieval_scores), softmax(lm_scores))
loss.backward()  → gradients flow to encoder parameters
optimizer.step() → retriever weights updated
```

**The key insight:** the generator's `P(response | document)` acts as a **teacher signal**. The retriever (student) learns to assign high similarity scores to documents that the generator finds useful. The KL divergence loss forces the retriever's ranking distribution to match the generator's utility distribution.

### 3.3 The Federated Training Round

Each federated round follows this sequence:

```
┌─────────────────────────────────────────────────────────────┐
│                     FEDERATED ROUND                         │
│                                                             │
│  Step 1: Server broadcasts global model weights             │
│          → set_weights(parameters) on each client           │
│                                                             │
│  Step 2: Each client trains locally (same LSR pipeline)     │
│          → DataCollatorForLSR → compute_loss() → LSRLoss    │
│          → backward() → optimizer.step()                    │
│          → Returns: (updated_weights, dataset_size, loss)   │
│                                                             │
│  Step 3: Server aggregates with FedAvg                      │
│          global_weight[i] = Σ(client_weight[i] × n_j) / Σ(n_j)│
│          (weighted by dataset size ONLY — no quality signal)│
│                                                             │
│  Step 4: Next round begins                                  │
└─────────────────────────────────────────────────────────────┘
```

**Currently, this aggregation is purely based on dataset size.** This is the gap our work targets.

### 3.4 Our Working Implementation

I have two working federated training scripts:

| Script | Approach | Status |
|--------|----------|--------|
| `federated_lsr.py` | Manual FedAvg loop (no Flower) | ✅ Working |
| `federated_lsr_flwr.py` | Flower `start_simulation()` with Ray | ✅ Working |

Both use 2 clients with disjoint data, 3 federated rounds, and demonstrate decreasing loss across rounds.

---

## 4. Work Done So Far — Fixing the Retriever Training Pipeline

### 4.1 The Problem

When I started, **retriever weights were not updating** during training. After training for multiple epochs, the model's weight hash remained identical — the optimizer was running but producing zero-effect updates.

### 4.2 Root Cause Analysis — Three Bugs

I identified and fixed three separate bugs that collectively prevented gradient flow from the loss function back to the retriever's encoder parameters:

#### Bug 1: Gradient Graph Severed in Data Collator

**Location:** `DataCollatorForLSR.__call__()` in `data_collators/huggingface/lsr.py`

**Problem:** The original data collator computed retrieval scores inside `__call__()` using `torch.tensor([score for score in ...], requires_grad=True)`. This created a **new leaf tensor** that was disconnected from the encoder's computation graph. Even though `requires_grad=True` was set, the tensor had no `grad_fn` linking it back to the encoder parameters — so `loss.backward()` had nothing to backpropagate through.

```
BEFORE (broken):
    retriever_scores = torch.tensor([n.score for n in source_nodes], requires_grad=True)
                       ↑ leaf tensor — no connection to encoder params!

    loss = KL_div(retriever_scores, lm_scores)
    loss.backward()  → gradients stop at the leaf tensor, never reach encoder
```

**Fix:** Removed all differentiable computation from the data collator. It now returns **raw data** (query strings, context text strings, and LM scores). The differentiable forward pass was moved to `compute_loss()` in the trainer.

#### Bug 2: DataLoader Prefetch Invalidated Computation Graph

**Problem:** Even after fixing Bug 1, a `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` occurred. This happened because the DataLoader **prefetches** the next batch while the current batch is being trained. If the differentiable forward pass runs in the collator, the prefetched batch's computation graph references encoder parameters that get modified in-place by `optimizer.step()` on the current batch — corrupting the graph.

```
BEFORE (broken):
    Batch N collator runs → builds computation graph through encoder params
    Batch N+1 collator runs (prefetch) → builds graph through SAME params
    Batch N training step → optimizer.step() modifies params IN-PLACE
    Batch N+1 training step → CRASH! Graph references stale params
```

**Fix:** Moving the differentiable forward pass to `compute_loss()` ensures the computation graph is built **inside `training_step()`**, one batch at a time, with no prefetch conflicts. Each batch builds and consumes its own graph before the next batch starts.

#### Bug 3: Optimizer Had Zero Parameters

**Location:** `LSRSentenceTransformerTrainer` in `trainers/huggingface/lsr.py`

**Problem:** `SentenceTransformerTrainer`'s parent class builds optimizer parameter groups by inspecting the **loss module's parameters** (designed for losses like `SoftmaxLoss` that have learnable weights). Since `LSRLoss` is pure KL divergence math with **no trainable parameters**, the optimizer was initialized with an empty parameter list. `optimizer.step()` was a no-op.

**Fix:** Overrode `create_optimizer()` to build parameter groups from `self.model.named_parameters()` (the actual encoder) instead of the loss module.

### 4.3 The Working Gradient Flow

After all three fixes, the gradient flow is:

```
compute_loss()
    model.forward(query) → query_embedding  [grad_fn ✅]
        ↓
    cosine_sim(query_emb, context_emb) → retrieval_scores  [grad_fn ✅]
        ↓
    KL_div(retrieval_scores, lm_scores) → loss  [grad_fn ✅]
        ↓
    loss.backward() → encoder.parameters() get .grad  ✅
        ↓
    optimizer.step() ← optimizer holds model params  ✅
        ↓
    Weights updated! ✅
```

### 4.4 Additional Fix: Training Loss Now Reported

**Location:** `HuggingFaceRAGTrainerManager._get_federated_trainer()` in `trainer_managers/huggingface.py`

The `train_wrapper` function that Flower calls during `client.fit()` was **discarding the actual training loss** and returning a hardcoded `TrainResult(loss=0)`. This meant:
- Per-client loss metrics were always zero
- No quality signal was available for aggregation

**Fix:** Changed both wrappers to capture and forward the real loss:

```diff
-_ = retriever_train_fn()
-return TrainResult(loss=0)
+result = retriever_train_fn()
+return TrainResult(loss=result.loss)
```

### 4.5 Verification Results

**Centralized training:**
- Weight hashes change after training (confirmed weights actually update)
- Loss decreases across epochs

**Federated training (Flower simulation):**
```
Round 1: aggregate_fit received 2 results, 0 failures — loss: 0.0279
Round 2: aggregate_fit received 2 results, 0 failures — loss: 0.0093
Round 3: aggregate_fit received 2 results, 0 failures — loss: 0.0036
```

---

## 5. Implementation Plan

### 5.1 Quality-Aware Federated Aggregation (Core Contribution)

**Goal:** Replace FedAvg's dataset-size-only weighting with quality-weighted aggregation.

**Approach:**

The training loss is now correctly reported per-client per-round (Section 4.4). I will use this loss — along with additional RAG-specific quality metrics — to weight each client's contribution during aggregation.

**What changes:**

1. **Custom FedAvg Strategy** — Subclass Flower's `FedAvg` and override `aggregate_fit()` to compute aggregation weights from a combination of:
   - Training loss (lower = better local fit)
   - Retrieval quality metrics (e.g., MRR, recall on a shared held-out set)

2. **Real Evaluation** — Replace the placeholder `test_fn()` with actual RAG evaluation that computes retrieval precision/recall on a validation set, giving the server a per-client quality score.

3. **Aggregation Formula** — Instead of pure dataset-size weighting:
   ```
   Current:  w_j = n_j / Σ(n_j)                   (dataset size only)
   Proposed: w_j = α·quality_j + (1-α)·(n_j/Σ(n_j))  (quality + size)
   ```

**Where it hooks in:** Flower's `Strategy.aggregate_fit()` receives per-client results (weights, dataset size, metrics). The metrics dict already carries `{"loss": actual_loss}` thanks to our fix. I add retrieval quality metrics and use them in the weighting formula.

### 5.2 Domain-Aware Selective Client Participation (Extended Contribution)

**Goal:** Only include clients whose data domain is relevant to the current training objective.

**Approach:**

Not all clients improve the global model equally. In a multi-domain federation (e.g., hospitals with different specialties), cardiology clients may actively harm a pulmonology-focused RAG retriever. Domain-aware selection addresses this.

**What changes:**

1. **Domain Profiling** — Characterize each client's data domain using embedding-space clustering or topic modeling over their knowledge store contents.

2. **Selective Participation** — Override Flower's `Strategy.configure_fit()` to select only domain-relevant clients for each round, based on similarity between client domain profiles and the target domain.

3. **Domain-Weighted Aggregation** — Combine with quality-aware aggregation: clients that are both high-quality AND domain-relevant get the highest weight.

**Where it hooks in:** Flower's `Strategy.configure_fit()` decides which clients participate in each round. Currently, all clients train every round. The override filters clients by domain relevance before sending them the global model.

---

## 6. Summary

| Phase | Status | Description |
|-------|--------|-------------|
| **Baseline — Fix training pipeline** | ✅ Done | Three gradient flow bugs fixed, centralized + federated training verified working |
| **Baseline — Flower integration** | ✅ Done | Working `fl.simulation.start_simulation()` script with actual loss reporting |
| **Core — Quality-aware aggregation** | 🔲 Next | Custom FedAvg strategy with quality-weighted client contributions |
| **Extended — Domain-aware selection** | 🔲 Planned | Selective client participation based on data domain relevance |
