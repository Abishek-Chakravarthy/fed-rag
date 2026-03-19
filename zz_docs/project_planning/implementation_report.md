# Implementation Report: Quality-Aware and Domain-Aware Federated RAG

---

## Table of Contents

1. [Objective 1: Quality-Aware Federated Aggregation](#objective-1-quality-aware-federated-aggregation)
   - 1.1 Theoretical Background
   - 1.2 The Current Baseline (What Happens Now)
   - 1.3 Proposed Algorithm
   - 1.4 Implementation Plan
2. [Objective 2: Domain-Aware Federated RAG with Selective Client Participation](#objective-2-domain-aware-federated-rag-with-selective-client-participation)
   - 2.1 Theoretical Background
   - 2.2 Proposed Algorithm
   - 2.3 Implementation Plan

---

# Objective 1: Quality-Aware Federated Aggregation

## 1.1 Theoretical Background

### Standard FedAvg

In Federated Averaging (McMahan et al., 2017), $K$ clients collaboratively train a global model $\theta$ without sharing data. Each round $t$:

1. Server broadcasts global weights $\theta^t$ to all clients
2. Each client $j$ trains locally on its own data $\mathcal{D}_j$ for $E$ epochs
3. Client $j$ sends updated weights $\theta_j^{t}$ to the server
4. Server aggregates:

$$\theta^{t+1} = \sum_{j=1}^{K} \frac{n_j}{N} \cdot \theta_j^{t}$$

Where $n_j = |\mathcal{D}_j|$ is client $j$'s dataset size and $N = \sum_{j=1}^K n_j$.

**The implicit assumption:** every training example is equally valuable. A client with more data should have more influence because it has "seen more" of the data distribution.

### Why This Fails for RAG

In RAG retriever training via LSR (Learned Sparse Retrieval), the training signal is the **KL divergence** between the retriever's document ranking and the generator's document utility:

$$\mathcal{L}_j^{LSR} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} D_{KL}\Big(\text{softmax}(\mathbf{s}_i^{LM}) \;\Big\|\; \text{softmax}(\mathbf{s}_i^{R})\Big)$$

Where:
- $\mathbf{s}_i^R \in \mathbb{R}^k$ = retriever's cosine similarities for query $i$: $s_{i,d}^R = \frac{\mathbf{q}_i \cdot \mathbf{c}_d}{\|\mathbf{q}_i\| \|\mathbf{c}_d\|}$
- $\mathbf{s}_i^{LM} \in \mathbb{R}^k$ = generator's utility scores: $s_{i,d}^{LM} = P_{LM}(\text{response}_i \mid \text{query}_i, \text{doc}_d)$
- $k$ = number of retrieved documents (`top_k`)

**The quality of this loss varies across clients.** Consider:

- **Client A** (hospital with well-curated medical records): Domain-relevant query-response pairs produce meaningful LM scores → retriever learns useful rankings → low final loss → high-quality weight update
- **Client B** (hospital with noisy OCR-scanned records): Garbage context texts produce near-random LM scores → retriever learns nothing useful → high final loss → low-quality weight update

Under standard FedAvg, if Client B has $2\times$ more data than Client A, Client B's noisy updates get $2\times$ the weight — actively degrading the global model.

---

## 1.2 The Current Baseline (What Happens Now)

### What the code does today

**File:** `fl_tasks/huggingface.py`, `FedAvg.aggregate_fit()` (Flower's built-in)

```python
# Flower's aggregate() function (from flwr/server/strategy/aggregate.py)

def aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    weighted_weights = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in results
    ]

    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
```

This is pure $\frac{n_j}{N}$ weighting — the only signal is `num_examples`, which comes from `len(self.train_dataset)` in the client's `fit()` method.

**What the client returns today:**

**File:** `fl_tasks/huggingface.py`, line 147–162

```python
def fit(self, parameters, config):
    self.set_weights(parameters)
    result: TrainResult = self.trainer(
        self.net, self.train_dataset, self.val_dataset, ...
    )
    return (
        self.get_weights(),          # updated local weights
        len(self.train_dataset),     # ← THIS is the only aggregation signal
        {"loss": result.loss},       # loss IS reported but NEVER used in aggregation
    )
```

The `{"loss": result.loss}` dict is passed to `fit_metrics_aggregation_fn` (if provided) for logging purposes only. **It is never used to influence the weight aggregation.**

**Where the strategy is created:**

**File:** `fl_tasks/huggingface.py`, line 253–278

```python
def server(self, strategy=None, client_manager=None, **kwargs):
    if strategy is None:
        model = kwargs.pop(self._trainer_spec.net_parameter)
        ndarrays = _get_weights(model)
        parameters = ndarrays_to_parameters(ndarrays)
        strategy = FedAvg(
            fraction_evaluate=1.0,
            initial_parameters=parameters,
        )
    return HuggingFaceFlowerServer(client_manager=client_manager, strategy=strategy)
```

**Key insight:** The `server()` method accepts a `strategy` parameter. If we pass our own custom strategy, it uses that instead of vanilla `FedAvg`. This is the **primary injection point**.

---

## 1.3 Proposed Algorithm

### Quality-Aware FedAvg (QA-FedAvg)

Replace the aggregation weight $w_j = \frac{n_j}{N}$ with:

$$w_j = \frac{\alpha \cdot q_j + (1-\alpha) \cdot \frac{n_j}{N}}{\sum_{m=1}^K \left[\alpha \cdot q_m + (1-\alpha) \cdot \frac{n_m}{N}\right]}$$

Where:
- $\alpha \in [0, 1]$ is a hyperparameter controlling quality vs. quantity trade-off
- $q_j$ is the **normalized quality score** of client $j$
- At $\alpha = 0$, we recover standard FedAvg
- At $\alpha = 1$, we use pure quality weighting

### Defining the Quality Score $q_j$

We define $q_j$ using the **inverse training loss**:

$$q_j = \frac{1 / (\mathcal{L}_j + \epsilon)}{\sum_{m=1}^{K} 1 / (\mathcal{L}_m + \epsilon)}$$

Where $\mathcal{L}_j$ is the LSR training loss reported by client $j$ at the end of this round, and $\epsilon$ is a small constant (e.g., $10^{-8}$) to avoid division by zero.

**Why inverse loss works:** Lower LSR loss means the retriever's ranking distribution is closer to the generator's utility distribution — i.e., the retriever has learned to rank documents the way the generator finds most helpful. This client's weight updates are more informative.

**Alternative quality score (evaluation-based):**

If a shared validation set is available, we can use retrieval quality metrics:

$$q_j = \text{MRR}_j(\mathcal{V}) = \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} \frac{1}{\text{rank}_{i,j}}$$

Where $\text{rank}_{i,j}$ is the rank of the first relevant document for query $i$ under client $j$'s updated model (evaluated on shared validation set $\mathcal{V}$).

### Per-Layer Aggregation

For each layer $l$ of the model:

$$\theta_l^{t+1} = \sum_{j=1}^{K} w_j \cdot \theta_{l,j}^{t}$$

Where $w_j$ is the quality-aware weight above (same weight for all layers).

---

## 1.4 Implementation Plan

### Step 1: Create `QualityAwareFedAvg` Strategy

**New file:** `src/fed_rag/strategies/quality_aware_fedavg.py`

```python
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np

class QualityAwareFedAvg(FedAvg):
    """FedAvg with quality-aware aggregation weights."""

    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.epsilon = epsilon

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # ── Extract per-client data ──
        client_data = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            n_examples = fit_res.num_examples
            loss = fit_res.metrics.get("loss", 1.0)  # default to 1.0 if missing
            client_data.append((ndarrays, n_examples, loss))

        # ── Compute quality scores (inverse loss) ──
        losses = np.array([loss for _, _, loss in client_data])
        inverse_losses = 1.0 / (losses + self.epsilon)
        quality_scores = inverse_losses / inverse_losses.sum()  # normalized q_j

        # ── Compute dataset-size weights (standard FedAvg) ──
        n_examples = np.array([n for _, n, _ in client_data], dtype=float)
        size_weights = n_examples / n_examples.sum()

        # ── Combine: w_j = α·q_j + (1-α)·(n_j/N) ──
        combined_weights = self.alpha * quality_scores + (1 - self.alpha) * size_weights
        combined_weights = combined_weights / combined_weights.sum()  # re-normalize

        # ── Weighted aggregation ──
        # For each layer: θ_l = Σ w_j · θ_{l,j}
        all_weights = [ndarrays for ndarrays, _, _ in client_data]
        num_layers = len(all_weights[0])

        aggregated = []
        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(all_weights[0][layer_idx])
            for j, client_weights in enumerate(all_weights):
                layer_sum += combined_weights[j] * client_weights[layer_idx]
            aggregated.append(layer_sum)

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        # ── Log metrics ──
        metrics = {
            "quality_weights": str(combined_weights.tolist()),
            "losses": str(losses.tolist()),
        }

        # Also run any user-provided metrics aggregation fn
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            user_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics.update(user_metrics)

        return parameters_aggregated, metrics
```

### Step 2: Modify the Flower Simulation Script

**Modified file:** `zz_coderuns/baseline/federated_lsr_flwr.py`

Replace:
```python
strategy = fl.server.strategy.FedAvg(...)
```

With:
```python
from fed_rag.strategies.quality_aware_fedavg import QualityAwareFedAvg

strategy = QualityAwareFedAvg(
    alpha=0.5,                      # tune this
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=initial_parameters,
    fit_metrics_aggregation_fn=weighted_average,
)
```

No changes needed to `client.fit()` — it already returns `{"loss": result.loss}` in the metrics dict, and our custom `aggregate_fit()` reads `fit_res.metrics.get("loss")`.

### Step 3: Implement Real Evaluation (Optional Enhancement)

**Modified file:** `trainer_managers/huggingface.py`

Replace the placeholder `test_fn`:

```python
# CURRENT (placeholder):
def test_fn(model: HFModelType, eval_dataset: Dataset) -> TestResult:
    return TestResult(loss=0.42, metrics={})

# PROPOSED (real evaluation):
def test_fn(model: HFModelType, eval_dataset: Dataset) -> TestResult:
    # Run retrieval on eval queries and compute MRR
    reciprocal_ranks = []
    for example in eval_dataset:
        query = example["query"]
        # ... encode query, retrieve, compute rank of relevant doc ...
        reciprocal_ranks.append(1.0 / rank)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return TestResult(loss=1.0 - mrr, metrics={"mrr": mrr})
```

### Step 4: Extend Client Metrics (Optional)

**Modified file:** `fl_tasks/huggingface.py`

Add retrieval quality metrics to `fit()`:

```python
def fit(self, parameters, config):
    self.set_weights(parameters)
    result = self.trainer(self.net, self.train_dataset, self.val_dataset, ...)
    return (
        self.get_weights(),
        len(self.train_dataset),
        {
            "loss": result.loss,
            # Additional RAG-specific metrics:
            # "mrr": self._compute_mrr_on_val_set(),
        },
    )
```

### File-Level Summary for Objective 1

| File | Action | What Changes |
|------|--------|-------------|
| `src/fed_rag/strategies/quality_aware_fedavg.py` | **NEW** | `QualityAwareFedAvg` class overriding `aggregate_fit()` |
| `zz_coderuns/baseline/federated_lsr_flwr.py` | **MODIFY** | Use `QualityAwareFedAvg` instead of `FedAvg` |
| `src/fed_rag/trainer_managers/huggingface.py` | **MODIFY** (optional) | Replace placeholder `test_fn` with real evaluation |
| `src/fed_rag/fl_tasks/huggingface.py` | **MODIFY** (optional) | Add additional quality metrics to `fit()` return |

---

---

# Objective 2: Domain-Aware Federated RAG with Selective Client Participation

## 2.1 Theoretical Background

### The Problem with Full Participation

In standard FedAvg (and even in QA-FedAvg from Objective 1), **all clients participate in every round**. This creates problems in heterogeneous federations:

**Example:** A federation of hospitals fine-tuning a medical RAG system:
- Client A: Cardiology records
- Client B: Oncology records  
- Client C: Radiology records

If the goal is to improve the RAG system for **cardiology queries**, clients B and C contribute weight updates trained on irrelevant data. Even quality-aware weighting cannot fully address this because:

1. Client B might have **low loss** on its oncology data (high quality within its domain) but its updates still push the retriever toward oncology-relevant document rankings — not what we want for cardiology.
2. Communication cost: Every client downloads global weights + uploads local weights every round, regardless of relevance.

### Formal Definition

Let each client $j$ have a data distribution $\mathcal{P}_j$ over a domain $\mathcal{D}_j$. The target domain is $\mathcal{D}^*$. We want to maximize performance on the target domain:

$$\min_\theta \; \mathbb{E}_{(x,y) \sim \mathcal{P}^*} \left[\mathcal{L}^{LSR}(\theta; x, y)\right]$$

But in federated learning, we can only observe:

$$\min_\theta \; \sum_{j=1}^K w_j \cdot \mathbb{E}_{(x,y) \sim \mathcal{P}_j} \left[\mathcal{L}^{LSR}(\theta; x, y)\right]$$

If $\mathcal{P}_j$ is very different from $\mathcal{P}^*$, including client $j$ in the sum adds noise to the gradient direction.

---

## 2.2 Proposed Algorithm

### Domain-Aware Selective FedAvg (DAS-FedAvg)

**Key idea:** Before each round, compute a **domain relevance score** for each client and only select clients above a threshold $\tau$.

### Domain Representation

Each client's domain is represented by the **centroid embedding** of its knowledge store:

$$\bar{\mathbf{e}}_j = \frac{1}{|S_j|}\sum_{d \in S_j} \mathbf{e}_d$$

Where $S_j$ is client $j$'s knowledge store and $\mathbf{e}_d$ is the embedding of document $d$ (already computed during indexing — these are the embeddings stored in `KnowledgeNode.embedding`).

The target domain is represented similarly, using a shared validation set $\mathcal{V}$:

$$\bar{\mathbf{e}}^* = \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} \mathbf{e}_{q_i}$$

Where $\mathbf{e}_{q_i}$ is the embedding of validation query $i$.

### Domain Relevance Score

$$d_j = \cos\left(\bar{\mathbf{e}}_j, \; \bar{\mathbf{e}}^*\right) = \frac{\bar{\mathbf{e}}_j \cdot \bar{\mathbf{e}}^*}{\|\bar{\mathbf{e}}_j\| \cdot \|\bar{\mathbf{e}}^*\|}$$

### Client Selection

For each round $t$, select clients:

$$S^t = \{j \mid d_j > \tau\}$$

Where $\tau \in [-1, 1]$ is the selection threshold (a hyperparameter). Alternatively, select the top-$p\%$ of clients by domain relevance:

$$S^t = \text{top-}p\%\left(\{d_j\}_{j=1}^K\right)$$

### Combined Aggregation

Aggregation applies only over selected clients, using quality-aware weights:

$$\theta^{t+1} = \sum_{j \in S^t} w_j \cdot \theta_j^{t}$$

Where $w_j$ is the quality-aware weight from Objective 1, re-normalized over the selected subset $S^t$:

$$w_j = \frac{\alpha \cdot q_j + (1-\alpha) \cdot \frac{n_j}{N_{S^t}}}{\sum_{m \in S^t} \left[\alpha \cdot q_m + (1-\alpha) \cdot \frac{n_m}{N_{S^t}}\right]}, \quad N_{S^t} = \sum_{m \in S^t} n_m$$

### Complete Round Procedure

```
Round t:
    1. Server computes domain relevance d_j for each available client j
    2. Server selects S^t = {j | d_j > τ}
    3. Server broadcasts θ^t to clients in S^t only
    4. Each selected client j ∈ S^t:
        a. Loads θ^t
        b. Trains locally → θ_j^t, reports loss L_j
    5. Server computes quality scores q_j from L_j
    6. Server aggregates:  θ^{t+1} = Σ_{j∈S^t} w_j · θ_j^t
```

---

## 2.3 Implementation Plan

### Step 1: Compute Domain Embeddings Per Client

Each client computes its domain centroid during initialization and reports it to the server.

**Modified file:** `fl_tasks/huggingface.py` — extend `HuggingFaceFlowerClient`

```python
class HuggingFaceFlowerClient(NumPyClient):
    def __init__(self, task_bundle, domain_embedding=None):
        super().__init__()
        self.task_bundle = task_bundle
        self.domain_embedding = domain_embedding  # np.ndarray, shape (embed_dim,)

    def get_properties(self, config):
        """Report domain embedding to the server."""
        if self.domain_embedding is not None:
            return {"domain_embedding": self.domain_embedding.tobytes()}
        return {}
```

**In the client factory (`client_fn`):**

```python
def client_fn(cid: str):
    knowledge_store, retriever = setup_knowledge_store()
    # ...

    # Compute domain centroid from knowledge store embeddings
    all_embeddings = [node.embedding for node in knowledge_store.all_nodes()]
    domain_embedding = np.mean(all_embeddings, axis=0)  # centroid

    flower_client = fl_task.client(model=model, train_dataset=..., val_dataset=...)
    flower_client.domain_embedding = domain_embedding  # attach domain info

    return flower_client.to_client()
```

### Step 2: Create `DomainAwareFedAvg` Strategy

**New file:** `src/fed_rag/strategies/domain_aware_fedavg.py`

This strategy extends `QualityAwareFedAvg` with selective client participation.

```python
from fed_rag.strategies.quality_aware_fedavg import QualityAwareFedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, Parameters
import numpy as np

class DomainAwareFedAvg(QualityAwareFedAvg):
    """QA-FedAvg with domain-aware client selection."""

    def __init__(
        self,
        target_embedding: np.ndarray,   # centroid of target domain
        tau: float = 0.0,               # domain relevance threshold
        alpha: float = 0.5,             # quality vs. size trade-off
        **kwargs,
    ):
        super().__init__(alpha=alpha, **kwargs)
        self.target_embedding = target_embedding / np.linalg.norm(target_embedding)
        self.tau = tau
        self.client_domain_embeddings = {}  # cid -> embedding

    def register_client_domain(self, cid: str, embedding: np.ndarray):
        """Register a client's domain embedding."""
        self.client_domain_embeddings[cid] = embedding / np.linalg.norm(embedding)

    def _domain_relevance(self, cid: str) -> float:
        """Compute cosine similarity between client domain and target domain."""
        if cid not in self.client_domain_embeddings:
            return 0.0  # unknown client → exclude
        return float(np.dot(
            self.client_domain_embeddings[cid],
            self.target_embedding
        ))

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple]:
        """Select only domain-relevant clients for this round."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample ALL available clients first
        all_clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=self.min_fit_clients,
        )

        # Filter by domain relevance
        selected = []
        for client in all_clients:
            relevance = self._domain_relevance(client.cid)
            if relevance > self.tau:
                selected.append((client, fit_ins))

        # Ensure minimum participation
        if len(selected) < self.min_fit_clients:
            # Fall back to top-k by relevance
            scored = [(client, self._domain_relevance(client.cid))
                      for client in all_clients]
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = [(client, fit_ins)
                        for client, _ in scored[:self.min_fit_clients]]

        return selected

    # aggregate_fit() is inherited from QualityAwareFedAvg
    # → quality-aware weighting applied only over selected clients
```

### Step 3: Target Domain Embedding

The target domain embedding $\bar{\mathbf{e}}^*$ is computed from the validation/evaluation queries:

```python
# In the simulation script:
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Compute target domain embedding from validation queries
val_queries = ["cardiology question 1", "cardiology question 2", ...]
val_embeddings = encoder.encode(val_queries)
target_embedding = np.mean(val_embeddings, axis=0)

strategy = DomainAwareFedAvg(
    target_embedding=target_embedding,
    tau=0.3,                        # relevance threshold
    alpha=0.5,                      # quality vs. size
    initial_parameters=initial_parameters,
    ...
)
```

### Step 4: Client Registration (Round 0 or via Properties)

Before the first training round, clients register their domain embeddings with the server. This can happen during `get_properties()` or during an initial handshake:

```python
# Option A: During client creation in client_fn(), set properties
# The strategy's on_fit_config_fn can request domain embeddings on round 1

# Option B: Pre-register in main() before start_simulation()
# Compute domain embeddings for all clients and pass to strategy
for cid in range(NUM_CLIENTS):
    ks, retriever = setup_client_knowledge_store(cid)
    embeddings = [node.embedding for node in ks.all_nodes()]
    centroid = np.mean(embeddings, axis=0)
    strategy.register_client_domain(str(cid), centroid)
```

### File-Level Summary for Objective 2

| File | Action | What Changes |
|------|--------|-------------|
| `src/fed_rag/strategies/domain_aware_fedavg.py` | **NEW** | `DomainAwareFedAvg` class overriding `configure_fit()` |
| `src/fed_rag/fl_tasks/huggingface.py` | **MODIFY** | Add `domain_embedding` to client, `get_properties()` returns it |
| `zz_coderuns/baseline/federated_lsr_flwr.py` | **MODIFY** | Use `DomainAwareFedAvg`, compute domain/target embeddings |

---

---

# Evaluation Plan

### Comparison Matrix

| Method | Aggregation Weight | Client Selection | Code |
|--------|-------------------|------------------|------|
| **FedAvg** (baseline) | $w_j = n_j / N$ | All clients | Existing |
| **QA-FedAvg** (Obj. 1) | $w_j = \alpha q_j + (1-\alpha)(n_j/N)$ | All clients | New strategy |
| **DAS-FedAvg** (Obj. 2) | QA-FedAvg over selected | $d_j > \tau$ | New strategy |
| **Centralized** (upper bound) | N/A | N/A | Existing |

### 6.1 Metrics and Formulas

To evaluate the success of our Federated RAG system, we utilize four primary metrics spanning the training and retrieval stages.

#### 1. LSR Training Loss (KL Divergence)

The core optimization objective for the retriever. We compute the **Kullback-Leibler (KL) Divergence** between the generator's document utility distribution and the retriever's ranking similarity distribution.

**Formula:**
$$\mathcal{L}_{LSR} = D_{KL}(P_{LM} \| P_{R}) = \sum_{i=1}^{k} P_{LM}(d_i|q) \log \frac{P_{LM}(d_i|q)}{P_{R}(d_i|q)}$$

*   **Explanation:** $P_{LM}$ is the target distribution from the generator (teacher), and $P_{R}$ is the retriever's current distribution (student). We apply a **softmax** to both the generator utility scores and the retriever cosine similarities to obtain these probability distributions.
*   **Why it's relevant:** KL divergence measures the information gain (or loss) when approximating one distribution with another. In RAG training, it essentially tells the retriever: *"Your document rankings should look exactly like what the generator finds most useful."*

#### 2. Mean Reciprocal Rank (MRR)

A standard information retrieval metric that focuses on how high the **first** relevant document appears in the retrieved results.

**Formula:**
$$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

*   **Explanation:** For each query $i$, $\text{rank}_i$ is the rank of the first relevant document correctly retrieved. If no relevant document is found in the top-k results, the reciprocal rank is 0.
*   **Why it's relevant:** RAG generation is most efficient and accurate when the most relevant context is provided at rank 1. MRR directly penalizes the model for pushing high-quality context further down the list, which might otherwise lead to its truncation or diluted importance in the LLM's prompt.

#### 3. Recall@k

Measures the system's ability to find **all** relevant material within a fixed-size window.

**Formula:**
$$\text{Recall@k} = \frac{|\text{relevant documents} \cap \text{top-}k \text{ results}|}{|\text{all relevant documents}|}$$

*   **Explanation:** It is the proportion of total relevant documents for a query that successfully appear in the first $k$ results.
*   **Why it's relevant:** Some queries have multiple relevant sources (e.g., medical claims from different experts). RAG systems need high Recall@k to ensure the generator has a comprehensive picture of the subject matter before constructing an answer.

#### 4. Normalized Discounted Cumulative Gain (NDCG@k)

A ranking metric that accounts for the relative importance (relevance) of multiple documents and applies a position-based penalty.

**Formula:**
$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \quad \text{DCG@k} = \sum_{r=1}^{k}\frac{2^{\text{rel}_r}-1}{\log_2(r+1)}$$

*   **Explanation:** DCG (Discounted Cumulative Gain) sums the relevance score $\text{rel}_r$ of each document at rank $r$, divided by a logarithmic discount term. IDCG is the Ideal DCG (the score if documents were sorted perfectly). NDCG normalizes the score between 0 and 1.
*   **Why it's relevant:** Unlike Recall/MRR which often treat relevance as binary (0 or 1), NDCG is essential when different documents have varying degrees of relevance. In medicine, a highly specific clinical trial result is more "relevant" than a general encyclopedia entry — NDCG rewards the model for prioritizing the "best" documents at the top.

### Experimental Setup

1. **Dataset:** Curate multi-domain QA pairs (e.g., medical subdomains)
2. **Clients:** 3–5 clients with controlled domain overlap
3. **Experiments:**
   - Vary $\alpha$ in $\{0, 0.25, 0.5, 0.75, 1.0\}$ for QA-FedAvg
   - Vary $\tau$ in $\{-0.5, 0, 0.3, 0.5, 0.7\}$ for DAS-FedAvg
   - Inject a "noisy client" (random data) to test robustness
   - Measure metrics at each round to plot convergence curves

---

# 7. Theoretical Convergence Goals

You're right—$\alpha$ and $\tau$ are not universal constants. If they were, we'd just hardcode them. Instead, we are trying to prove that our **adaptive mechanisms** change the *nature* of the convergence curve compared to the static FedAvg baseline.

We are specifically targeting three types of convergence improvement:

### 7.1 Convergence Rate (Efficiency)
Standard FedAvg in a heterogeneous (Non-IID) environment often suffers from **Client Drift**. Clients "pull" the global model in different directions, causing the global loss to jitter or converge very slowly.
- **Proof Goal:** Our method reaches a target retrieval quality (e.g., MRR = 0.6) in **$N$ fewer rounds** than standard FedAvg.
- **The Math:** We expect $\frac{\partial \text{MRR}}{\partial t}$ (the rate of improvement per round) to be higher with QA-FedAvg because we are dampening noisy updates.

### 7.2 Convergence Plateau (Accuracy)
In Non-IID settings, standard FedAvg often plateaus at a lower accuracy than centralized training (the "FL Gap").
- **Proof Goal:** Our method reaches a **higher final MRR/Recall** than standard FedAvg, effectively "closing the gap" with the centralized upper bound.
- **The Reality:** Standard FedAvg is a "lowest common denominator" approach. By using Quality-Aware aggregation, we allow high-quality clients to "pull" the model to higher performance levels that would otherwise be averaged out.

### 7.3 Convergence Stability (Robustness)
One "poisonous" or extremely noisy client can cause standard FedAvg to never converge at all.
- **Proof Goal:** Even with $K$ noisy/random clients injected into the federation, our system maintains a **monotonically decreasing loss curve**, while standard FedAvg becomes unstable or diverges.

---

# 8. Hyperparameter Sensitivity and Tuning

### 8.1 The "Brutal" Reality of $\alpha$ and $\tau$
You correctly pointed out that these change per dataset. This is why our final paper/report won't just say "use $\alpha=0.5$," but will instead show a **Sensitivity Analysis**:

- **$\alpha$ (Alpha) Sensitivity:** We will plot "Final MRR vs Alpha." 
  - If the curve peaks at $\alpha=0.6$, it proves that quality-weighting matters. 
  - If the curve is highest at $\alpha=0$, our idea failed (standard FedAvg won).
- **$\tau$ (Tau) Sensitivity:** We will plot "Domain Accuracy vs Tau."
  - This shows the trade-off between **Generalization** (low $\tau$, many clients) and **Specialization** (high $\tau$, few relevant clients).

### 8.2 Automated Tuning (Future Extension)
In a production system, $\alpha$ and $\tau$ shouldn't be guessed. They could be learned dynamically:
- **Dynamic $\alpha$:** Scale $\alpha$ as a function of the variance in reported losses. If losses are similar, keep $\alpha$ low; if one client has a massive loss, spike $\alpha$ to dampen its influence.
- **Dynamic $\tau$:** Scale $\tau$ based on the "Domain Diversity" of the current query batch.

---

# 9. Summary Table: What are we proving?

| Metric | What we want to see | Success Criteria |
|--------|---------------------|------------------|
| **Speed** | Steeper MRR curve | Fewer rounds to reach "useful" RAG. |
| **Peak** | Higher final MRR | Better end-to-end answers compared to baseline. |
| **Resilience** | Smooth loss curve with noise | System doesn't break when a "bad" client joins. |
| **Efficiency** | Lower cumulative bandwidth | Better performance with fewer client-server rounds. |

---

# 10. Recommended Datasets

To prove the value of quality-aware and domain-aware mechanisms, we need a dataset that is naturally **heterogeneous** (different domains) and **variable in quality**.

### 10.1 The Primary Choice: BEIR (Benchmarking Information Retrieval)

BEIR is a collection of diverse retrieval datasets. This is the best choice for your research because it's the gold standard for RAG benchmarking.

I recommend selecting **3-4 datasets** from the BEIR suite to represent different "Hospitals" or "Clients":

| Dataset Name | Domain | Scale (Docs) | Why use it? |
|--------------|--------|--------------|-------------|
| **NFCorpus** | Medical / Nutrition | 5,371 | Perfect for your hospital use case. |
| **SciFact** | Scientific Claims | 5,183 | High-quality, evidence-based domain. |
| **FiQA-2018** | Finance | 57,638 | Extremely different terminology (domain swap). |
| **ArguAna** | Argumentation | 8,674 | Tests ranking logic on non-factoid queries. |

### 10.2 How to use them for Objective 1 (Quality-Aware)
Take **NFCorpus** and split it between two clients:
- **Client 1 (Clean):** Standard NFCorpus query-response pairs.
- **Client 2 (Noisy):** Corrupt 50% of the query-response mapping or shuffle the documents.
- **Success Link:** Our QA-FedAvg should detect the higher loss in Client 2 and automatically reduce its influence on the global medical retriever.

### 10.3 How to use them for Objective 2 (Domain-Aware)
Assign different BEIR datasets to different clients:
- **Client A:** NFCorpus (Medical)
- **Client B:** SciFact (Science)
- **Client C:** FiQA (Finance)
- **Evaluation:** Set the "Target Domain" to **Medical**.
- **Success Link:** Our DAS-FedAvg should correctly calculate high cosine similarity for Client A and exclude Client C from the aggregation, preventing "domain dilution."

### 10.4 Implementation Note: Loading BEIR
You can load these easily via the `beir` library or directly from HuggingFace `datasets`:
```python
from datasets import load_dataset
# Example: Load Medical Nutrition corpus
corpus = load_dataset("Beir/nfcorpus", "corpus")
queries = load_dataset("Beir/nfcorpus", "queries")
qrels = load_dataset("Beir/nfcorpus", "qrels")
```
