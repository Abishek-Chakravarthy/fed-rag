Metrics We're Targeting
There are three layers of metrics:

Layer 1 — Training Objective (what the optimizer minimizes):

The LSR loss per client $j$ is the KL divergence between the retriever's document ranking and the generator's document utility:

$$\mathcal{L}j^{LSR} = \frac{1}{|B|}\sum{i \in B} \text{KL}\Big(\log\text{softmax}(\mathbf{s}_i^R) ;|; \text{softmax}(\mathbf{s}_i^{LM})\Big)$$

Where for each query $i$ in batch $B$:

$\mathbf{s}i^R \in \mathbb{R}^k$ = retriever's cosine similarity scores for top-$k$ documents: $s{i,d}^R = \cos(\mathbf{q}_i, \mathbf{c}_d)$
$\mathbf{s}i^{LM} \in \mathbb{R}^k$ = generator's utility scores: $s{i,d}^{LM} = P_\text{LM}(\text{response} \mid \text{query}, \text{doc}_d)$
The federated objective is to minimize the global loss:

$$\min_\theta ; \mathcal{L}\text{global}(\theta) = \sum{j=1}^{K} w_j \cdot \mathcal{L}_j(\theta)$$

Currently $w_j = \frac{n_j}{\sum n_k}$ (pure dataset size). Both contributions change how $w_j$ is computed.

Layer 2 — Retrieval Quality (what we evaluate the retriever on):

After training, we measure retrieval quality on a held-out evaluation set:

Metric	Formula	What it measures
MRR (Mean Reciprocal Rank)	$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$	How high the first relevant doc ranks. 1.0 = always rank 1.
Recall@k	$\text{Recall}@k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$	What fraction of all relevant docs appear in the top-$k$.
NDCG@k	$\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$, where $\text{DCG}@k = \sum_{i=1}^{k}\frac{2^{\text{rel}_i}-1}{\log_2(i+1)}$	Ranking quality — penalizes relevant docs appearing lower.
These are the primary comparison metrics — we compare quality-aware FedAvg vs. standard FedAvg vs. centralized training across these.

Layer 3 — End-to-End RAG Quality (downstream task):

Metric	What it measures
Answer Correctness	Semantic similarity (or exact match) between generated answer and ground truth
Faithfulness	Whether the generated answer is grounded in the retrieved documents (not hallucinated)
These are secondary metrics — they show that better retrieval actually translates to better final answers.

How Each Contribution Improves These Metrics
Idea 1 — Quality-Aware Aggregation:

Current FedAvg: $$w_j = \frac{n_j}{\sum_{k=1}^K n_k}$$

Proposed quality-aware weighting: $$w_j = \frac{\alpha \cdot q_j + (1-\alpha) \cdot \frac{n_j}{\sum n_k}}{\sum_{m=1}^K \Big[\alpha \cdot q_m + (1-\alpha) \cdot \frac{n_m}{\sum n_k}\Big]}$$

Where $q_j$ is the quality score of client $j$, derived from:

Option A (inverse loss): $q_j = \frac{1/\mathcal{L}_j}{\sum_k 1/\mathcal{L}_k}$ — lower loss → higher weight
Option B (eval-based): $q_j = \text{MRR}_j$ on a shared validation set — better retrieval quality → higher weight
The hyperparameter $\alpha \in [0,1]$ controls how much quality matters vs. dataset size. At $\alpha=0$ we recover standard FedAvg.

What we expect to improve: Convergence speed (fewer rounds to reach target MRR), final MRR/Recall@k/NDCG@k, and robustness to noisy/low-quality clients.

Idea 2 — Domain-Aware Selection:

For each round $t$, instead of selecting all $K$ clients, select a subset $S^t \subseteq {1, \ldots, K}$ based on domain relevance:

$$S^t = {j \mid d_j > \tau}$$

Where $d_j$ is the domain similarity between client $j$'s data and the target domain. This can be computed as:

$$d_j = \cos\Big(\bar{\mathbf{e}}_j^{\text{client}}, ; \bar{\mathbf{e}}^{\text{target}}\Big)$$

Where $\bar{\mathbf{e}}_j^{\text{client}}$ is the mean embedding of client $j$'s knowledge store, and $\bar{\mathbf{e}}^{\text{target}}$ is the mean embedding of the target domain's validation set.

Aggregation then applies only over selected clients:

$$\theta^{t+1} = \sum_{j \in S^t} w_j \cdot \theta_j^t, \quad \text{where } w_j \text{ uses quality-aware weighting from Idea 1}$$

What we expect to improve: Same retrieval metrics (MRR, Recall@k, NDCG@k) but specifically on target-domain queries, plus better communication efficiency (fewer clients → fewer weight transfers per round).

In summary — the comparison table we'd produce in the paper:

Method	Aggregation Weight $w_j$	Client Selection	Metrics Evaluated
FedAvg (baseline)	$n_j / \sum n_k$	All clients	MRR, Recall@k, NDCG@k
Quality-Aware (ours)	$\alpha \cdot q_j + (1-\alpha) \cdot n_j/N$	All clients	MRR, Recall@k, NDCG@k
Domain-Aware (ours)	Quality-aware over selected	$d_j > \tau$ only	Domain MRR, Recall@k, NDCG@k
Centralized (upper bound)	N/A	N/A (all data pooled)	MRR, Recall@k, NDCG@k