# 02 — Methodology & Mathematical Formulation

## 1. LSR (LM-Supervised Retrieval) Training
LSR is the training method used to fine-tune the **retriever** component of the RAG system. It uses the **generator (LLM)** as a frozen teacher signal.

### Core Idea
For a training pair (query $x$, ground-truth response $y$), the retriever retrieves top-$k$ documents $\{d_1, ..., d_k\}$ from the knowledge store. The generator scores each document by computing $P(y \mid x, d_i)$ — i.e., how useful is document $d_i$ for producing the correct response? The retriever then learns to assign high similarity scores to exactly those documents the generator found useful.

### Loss Function
The LSR loss for client $j$ on batch $B$ is the KL-Divergence between the retriever's ranking distribution and the generator's utility distribution:

$$\mathcal{L}_j^{LSR} = \frac{1}{|B|}\sum_{i \in B} \text{KL}\Big(\log\text{softmax}(\mathbf{s}_i^R) \;\|\; \text{softmax}(\mathbf{s}_i^{LM})\Big)$$

Where:
- $\mathbf{s}_i^R \in \mathbb{R}^k$ = retriever's cosine similarity scores: $s_{i,d}^R = \cos(\mathbf{q}_i, \mathbf{c}_d)$
- $\mathbf{s}_i^{LM} \in \mathbb{R}^k$ = generator's utility scores: $s_{i,d}^{LM} = P_{LM}(\text{response} \mid \text{query}, d)$

### Two-Stage Pipeline
**Stage A — Data Collator (non-differentiable):** For each (query, response), retrieve top-k documents and score them with the frozen generator. Returns raw data: query strings, context texts, generator scores.

**Stage B — compute_loss() (fully differentiable):** Encodes query through the model's forward pass (preserving the gradient graph), computes cosine similarity against context embeddings, and computes KL divergence loss. Gradients flow back to the encoder parameters.

This two-stage design is critical — it prevents DataLoader prefetching from corrupting the computation graph (see bug fix details in file 03).

---

## 2. Standard FedAvg
In standard Federated Averaging (McMahan et al., 2017), the global model after round $t$ is:

$$\theta^{t+1} = \sum_{j=1}^{K} w_j \cdot \theta_j^t, \quad w_j = \frac{n_j}{\sum_{k=1}^K n_k}$$

Where $\theta_j^t$ is client $j$'s updated model after local training, and $n_j$ is the number of training examples on client $j$. This weights clients purely by dataset size.

---

## 3. QA-FedAvg (Quality-Aware Federated Aggregation)
We propose replacing the pure data-size weight with a combination of data size and **training quality**:

### Step 1: Compute Quality Scores
The quality score for client $j$ is the normalized inverse of its training loss $\mathcal{L}_j$:

$$q_j = \frac{1 / (\mathcal{L}_j + \epsilon)}{\sum_{k=1}^K 1 / (\mathcal{L}_k + \epsilon)}$$

Where $\epsilon = 10^{-8}$ prevents division by zero. Lower loss → higher quality score.

### Step 2: Combined Aggregation Weight
The final weight blends quality and size using hyperparameter $\alpha \in [0, 1]$:

$$w_j = \frac{\alpha \cdot q_j + (1 - \alpha) \cdot \frac{n_j}{N}}{\sum_{m=1}^K \left[ \alpha \cdot q_m + (1 - \alpha) \cdot \frac{n_m}{N} \right]}$$

Where $N = \sum_{k=1}^K n_k$.

### Properties
| α Value | Behavior | Use Case |
|---------|----------|----------|
| α = 0.0 | Pure FedAvg (dataset size only) | Baseline / homogeneous data |
| α = 0.6 | Optimal balance (validated) | Heterogeneous data quality |
| α = 1.0 | Pure quality weighting | Risk of over-correction |

### Why α=0.6 is Optimal
- At α=0, noisy clients receive full weight, dragging down the global model.
- At α=1, quality estimates from a single training epoch can be noisy themselves, leading to over-correction.
- At α=0.6, the aggregation correctly penalizes low-quality clients while maintaining the stabilizing effect of data-size proportional weighting.

---

## 4. Evaluation Metrics

### Layer 1: Training Loss (Per-Round)
The KL divergence LSR loss, averaged across all training examples. This is the primary optimization target and the quality signal used for QA-FedAvg.

### Layer 2: Retrieval Quality (Pre/Post Training)
Evaluated on a held-out evaluation set of (query, relevant_doc_id) pairs:

- **MRR (Mean Reciprocal Rank)**: $\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$ — How high the first relevant document ranks (1.0 = always rank 1).
- **Recall@k**: $\frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$ — What fraction of relevant documents appear in the top-k results.
- **NDCG@k**: $\frac{\text{DCG}@k}{\text{IDCG}@k}$ where $\text{DCG}@k = \sum_{i=1}^{k}\frac{2^{\text{rel}_i}-1}{\log_2(i+1)}$ — Ranking quality with position-based penalties.
