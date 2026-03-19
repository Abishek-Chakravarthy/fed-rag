# 03 — Implementation Details & Bug Fixes

## 1. Technology Stack
| Component | Technology | Details |
|-----------|-----------|---------|
| Retriever | `sentence-transformers/all-MiniLM-L6-v2` | 22M params, 384-dim embeddings |
| Generator | `distilgpt2` | 82M params, frozen during retriever training |
| Knowledge Store | `InMemoryKnowledgeStore` (fed-rag) | Pre-embedded document vectors |
| FL Framework | Flower (`flwr`) | `start_simulation()` with Ray backend |
| RAG Library | `fed-rag` (open-source) | https://github.com/nerdai/fed-rag |
| Dataset | BEIR Benchmark (NFCorpus, SciFact) | Standard IR evaluation datasets |
| Hardware | Apple M1 Pro, 16GB RAM | CPU-only training |

## 2. Experimental Configuration
| Parameter | Value |
|-----------|-------|
| Num Clients | 3 |
| Num Rounds | 3 |
| Batch Size | 8 |
| Learning Rate | 2e-6 |
| Max Training Pairs | 500 (split across clients) |
| Max Eval Pairs | 100 |
| Max Corpus Documents | 1000 |
| Retriever top-k | 10 |
| Optimizer | AdamW (warmup_ratio=0.1, weight_decay=0.01) |
| Random Seed | 42 |

## 3. Critical Bug Fixes in FedRAG Library
When we first attempted to train the retriever using the FedRAG library, **model weights did not update at all** — the weight hash remained identical before and after training. We traced this to three separate bugs that collectively broke the gradient pipeline:

### Bug 1: Gradient Graph Severed in Data Collator
**File:** `src/fed_rag/data_collators/huggingface/lsr.py`

**Problem:** The original `DataCollatorForLSR.__call__()` computed retrieval scores inside the collator using `torch.tensor([score for score in ...], requires_grad=True)`. This created a **new leaf tensor** disconnected from the encoder's computation graph. Even though `requires_grad=True` was set, the tensor had no `grad_fn` linking it to encoder parameters, so `loss.backward()` had nowhere to propagate gradients.

**Fix:** Removed all differentiable computation from the data collator. It now returns only **raw data** (query strings, context text strings, generator scores as detached tensors). The differentiable forward pass was moved to `compute_loss()` in the trainer, where `model.forward()` is called to build a proper computation graph.

### Bug 2: DataLoader Prefetch Invalidated Computation Graph
**File:** `src/fed_rag/data_collators/huggingface/lsr.py`

**Problem:** PyTorch DataLoaders prefetch the next batch while the current batch is being trained. If the differentiable forward pass runs in the collator, the prefetched batch's computation graph references encoder parameters that get modified in-place by `optimizer.step()` on the current batch, causing `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.

**Fix:** Moving the forward pass to `compute_loss()` ensures the graph is built **inside `training_step()`**, one batch at a time, with no prefetch conflicts. Each batch builds and consumes its own independent computation graph.

### Bug 3: Optimizer Initialized with Zero Parameters
**File:** `src/fed_rag/trainers/huggingface/lsr.py`

**Problem:** `SentenceTransformerTrainer`'s parent class constructs optimizer parameter groups by inspecting the **loss module's parameters** (designed for losses like `SoftmaxLoss` that have learnable weights). Since `LSRLoss` is pure KL divergence with **no trainable parameters**, the optimizer's parameter list was empty. `optimizer.step()` was literally a no-op.

**Fix:** Overrode `create_optimizer()` in `LSRSentenceTransformerTrainer` to build parameter groups from `self.model.named_parameters()` (the actual encoder weights) instead of the loss module. This ensures AdamW operates on the retriever's ~22M parameters.

### Bug 4: Training Loss Not Reported to Server
**File:** `src/fed_rag/trainer_managers/huggingface.py`

**Problem:** The `train_wrapper` function called by Flower during `client.fit()` was discarding the actual training loss: `_ = retriever_train_fn(); return TrainResult(loss=0)`. This meant the server always received `loss=0` for every client, making quality-aware aggregation impossible.

**Fix:** Changed to: `result = retriever_train_fn(); return TrainResult(loss=result.loss)`. Now the real per-client training loss is propagated to the server's `aggregate_fit()`.

### Verified Gradient Flow (After Fixes)
```
compute_loss():
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

## 4. QA-FedAvg Strategy Implementation
We implemented `QualityAwareFedAvg` as a custom Flower `Strategy` class (~80 lines) that subclasses `FedAvg` and overrides `aggregate_fit()`. The key implementation steps:

1. Extract per-client `(weights, num_examples, loss)` from `FitRes` objects
2. Compute inverse-loss quality scores: `q_j = (1/L_j) / Σ(1/L_k)`
3. Compute data-size weights: `s_j = n_j / N`
4. Combine: `w_j = α·q_j + (1-α)·s_j`, then normalize
5. Per-layer weighted aggregation of model parameters
6. Store diagnostics (losses, scores, weights) for CSV logging

## 5. Data Preparation Pipeline
**File:** `prepare_beir_data.py`

1. **Load BEIR dataset**: Downloads corpus, queries, and relevance judgments (qrels) from HuggingFace
2. **Build (query, response) pairs**: For each positive relevance judgment (score > 0), creates a training pair where the "response" is the relevant document text (truncated to 500 characters)
3. **Build Knowledge Store**: Embeds corpus documents using the retriever's context encoder and stores them as `KnowledgeNode` objects in `InMemoryKnowledgeStore`
4. **Split data**: IID split (random shuffle + equal partition) or Non-IID split (different datasets per client)
5. **Evaluation**: Computes MRR, Recall@k, and NDCG@k by encoding queries, retrieving top-k from the store, and comparing against ground-truth relevant document IDs
