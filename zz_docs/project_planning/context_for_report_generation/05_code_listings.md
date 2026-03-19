# 05 — Key Code Listings

## 1. QualityAwareFedAvg Strategy (Complete Implementation)
**File:** `zz_coderuns/quality_aware_fedrag/quality_aware_fedavg.py`

```python
class QualityAwareFedAvg(FedAvg):
    """FedAvg with quality-aware aggregation weights.
    
    Toggle:
        alpha=0.0  → identical to standard FedAvg
        alpha>0.0  → quality-awareness proportional to alpha
    """

    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.round_quality_info: list[dict] = []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Extract per-client data
        client_data = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            n_examples = fit_res.num_examples
            loss = fit_res.metrics.get("loss", 1.0)
            client_data.append((ndarrays, n_examples, loss))

        # Compute quality scores q_j = (1/L_j) / Σ(1/L_m)
        losses = np.array([loss for _, _, loss in client_data])
        inverse_losses = 1.0 / (losses + self.epsilon)
        quality_scores = inverse_losses / inverse_losses.sum()

        # Compute dataset-size weights (standard FedAvg)
        n_examples = np.array([n for _, n, _ in client_data], dtype=float)
        size_weights = n_examples / n_examples.sum()

        # Combine: w_j = α·q_j + (1-α)·(n_j/N)
        combined_weights = (
            self.alpha * quality_scores
            + (1 - self.alpha) * size_weights
        )
        combined_weights = combined_weights / combined_weights.sum()

        # Per-layer weighted aggregation
        all_weights = [ndarrays for ndarrays, _, _ in client_data]
        num_layers = len(all_weights[0])
        aggregated = []
        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(all_weights[0][layer_idx])
            for j, client_weights in enumerate(all_weights):
                layer_sum += combined_weights[j] * client_weights[layer_idx]
            aggregated.append(layer_sum)

        return ndarrays_to_parameters(aggregated), metrics_aggregated
```

## 2. compute_loss() — Differentiable Forward Pass (Bug Fix #1 & #2)
**File:** `src/fed_rag/trainers/huggingface/lsr.py`

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """The differentiable forward pass happens HERE (not in the data collator)."""
    queries = inputs["queries"]
    context_texts_batch = inputs["context_texts"]
    lm_scores = inputs["lm_scores"]

    batch_retriever_scores = []
    for query, context_texts in zip(queries, context_texts_batch):
        # Query embedding via model forward (preserves grad graph)
        query_features = model.tokenize([query])
        query_features = {k: v.to(model.device) for k, v in query_features.items()}
        query_embedding = model(query_features)["sentence_embedding"]

        # Context embeddings (no grad needed — stored docs)
        with torch.no_grad():
            context_embedding = model.encode(context_texts, convert_to_tensor=True)

        # Cosine similarity (differentiable w.r.t. query_embedding)
        query_norm = F.normalize(query_embedding, p=2, dim=1)
        context_norm = F.normalize(context_embedding, p=2, dim=1)
        retriever_scores = torch.mm(query_norm, context_norm.t()).squeeze(0)
        batch_retriever_scores.append(retriever_scores)

    retrieval_scores = torch.stack(batch_retriever_scores, dim=0)
    loss = self.loss(retrieval_scores, lm_scores)  # KL divergence
    return (loss, inputs) if return_outputs else loss
```

## 3. create_optimizer() — Bug Fix #3
**File:** `src/fed_rag/trainers/huggingface/lsr.py`

```python
def create_optimizer(self):
    """Override: build optimizer from model params, not loss module params."""
    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters()
                        if n in decay_parameters and p.requires_grad],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters()
                        if n not in decay_parameters and p.requires_grad],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon)
    return self.optimizer
```

## 4. train_wrapper — Bug Fix #4 (Loss Forwarding)
**File:** `src/fed_rag/trainer_managers/huggingface.py`

```diff
# BEFORE (broken):
-def train_wrapper(model, train_dataset, val_dataset):
-    _ = retriever_train_fn()
-    return TrainResult(loss=0)  # ← real loss discarded!

# AFTER (fixed):
+def train_wrapper(model, train_dataset, val_dataset):
+    result = retriever_train_fn()
+    return TrainResult(loss=result.loss)  # ← real loss forwarded to server
```
