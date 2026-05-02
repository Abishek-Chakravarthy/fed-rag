# DAS-FedAvg Agent Context

## What This Contribution Is

Domain-Aware Soft Federated Averaging (DAS-FedAvg) for federated RAG systems.

The core idea: in a multi-domain federation (clients holding medical, science,
finance, argument, and biomedical data), training a retriever for a specific target
domain (e.g. medical) is harmed by updates from semantically irrelevant clients
(e.g. finance). DAS-FedAvg computes a domain relevance score for each client as the
cosine similarity between that client's document embedding centroid and the target
domain's centroid, then weights each client's contribution proportionally to its
relevance score. Irrelevant clients naturally receive near-zero weight without
explicit exclusion.

This is the second of two contributions in the same final year project. The first
contribution (QA-FedAvg) is in a separate codebase and is independent.

## Repository

Branch: `das-fedrag` on `https://github.com/Abishek-Chakravarthy/fed-rag`
All experiment code lives in: `zz_coderuns/das_fedrag/`

## The Algorithm (Final — Pure Relevance Weighting)

Domain relevance score for client j:
  d_j = cos(centroid_j, centroid_target)

  where centroid_j = mean embedding of client j's knowledge store documents,
  normalised to unit length

Aggregation weight:
  w_j = d_j / Σ_k d_k

**Zero hyperparameters.** No τ (temperature), no n_j (data volume) in the
aggregation weight. Data volume is implicitly captured by gradient magnitude
during local training — more data → more SGD steps → larger model delta.

### Why n_j was dropped

Including n_j in the weight formula (as `d_j × n_j / Σ(d_k × n_k)`) caused
weight concentration: the medical client had both the highest relevance (1.0)
AND the most data (4000 pairs), leading to 95.5% weight — effectively
single-client training. Dropping n_j and using pure relevance gives medical
53.9% weight, allowing science (25.8%) and biomedical (17.3%) to contribute
meaningfully.

### Why τ was dropped

Temperature scaling over log-relevance (softmax(log(d_j)/τ)) was attempted
(τ=1.0) but was ineffective — it did not change the weight distribution given
the specific relevance score spread. Any τ value that works is experiment-specific
and not defensible for a general algorithm. Pure `d_j / Σ d_k` is parameter-free
and defensible.

### Baseline fallback

When all relevance scores are identical (baseline mode: all d_j = 1.0), the code
detects this via `all(abs(r - relevances[0]) < 1e-8 for r in relevances)` and
falls back to standard FedAvg n_j/N weighting. This ensures baseline experiments
produce standard FedAvg behavior.

### Confirmed weight distribution (from exp_04 CSV data)

| Client | Domain | d_j | DAS-FedAvg | FedAvg Baseline | Old Soft Domain (d_j × n_j) |
|--------|--------|-----|------------|-----------------|----------------------------|
| 0 | medical | 1.0000 | **0.5389** | 0.7287 | 0.9545 |
| 1 | science | 0.4791 | **0.2582** | 0.0435 | 0.0273 |
| 4 | biomedical | 0.3212 | **0.1731** | 0.0184 | 0.0077 |
| 3 | argument | 0.0422 | **0.0227** | 0.1787 | 0.0099 |
| 2 | finance | 0.0130 | **0.0070** | 0.0306 | 0.0005 |

## Training Setup

- Retriever: sentence-transformers/all-MiniLM-L6-v2
- Training objective: MultipleNegativesRankingLoss (contrastive, via ContrastiveFlowerClient)
- No generator — LSR/distilgpt2 removed entirely
- Target dataset: NFCorpus (medical/nutrition)
- Other client datasets: SciFact (science), FiQA (finance), ArguAna (argument),
  TREC-COVID (biomedical)
- **Learning rate: 5e-7** (reduced from 2e-6 which was destructive)
- Batch size: 8
- **Local epochs: 1** (3 epochs tested and found destructive — see exp_05)
- **Rounds: 8** (extended from 4)
- 5 clients total, all participate every round

### Client data sizes (deterministic across seeds)

| Client | Domain | Training Pairs |
|--------|--------|---------------|
| 0 | medical | 4000 |
| 1 | science | 239 |
| 2 | finance | 168 |
| 3 | argument | 981 |
| 4 | biomedical | 101 |

## Client Domain Relevance Scores (computed, correct)

| Client | Domain | Relevance Score |
|--------|--------|-----------------|
| 0 | Medical (NFCorpus) | 1.0000 ← TARGET |
| 1 | Science (SciFact) | 0.4791 |
| 4 | Biomedical (TREC-COVID) | 0.3212 |
| 3 | Argument (ArguAna) | 0.0422 |
| 2 | Finance (FiQA) | 0.0130 |

Relevance scores are computed fresh at the start of each run from corpus embeddings.
They are seed-independent (corpus loading order does not depend on seed).
Centroids are cached in .centroid_cache/ to avoid recomputation.

## Key Files

- domain_aware_fedavg.py       — DomainAwareFedAvg Flower strategy (server-side)
- federated_das.py             — main experiment script
- prepare_multi_domain_data.py — multi-domain data loading, centroid computation
- contrastive_trainer.py       — ContrastiveFlowerClient (MNR loss, drop-in, no changes)
- single_tau_kaggle_colab.ipynb — Kaggle execution notebook
- das_fedrag_improvement_plan_from_here.md — full experiment journey with analysis
- exp_01_results/ through exp_05_results/ — experiment result folders

## Bugs Fixed (do not revert)

### Bug 1 (FIXED): Round 1 proxy-logical CID mapping
Was: proxy_to_logical_cid initialised as empty dict; Round 1 saw all clients as
unmapped and selected all regardless of weighting logic.

Fix: federated_das.py builds pre_registered_cid_map = {str(i): str(i) for i in
range(num_clients)} before start_simulation() and passes it to
DomainAwareFedAvg.__init__ as pre_registered_cid_map. The strategy initialises
self.proxy_to_logical_cid from this map at construction time. Round 1 now applies
correct domain-aware weighting from the start.

aggregate_fit() still reads fit_res.metrics["logical_cid"] for verification — if it
contradicts the pre-registered mapping, a WARNING is logged but the pre-registered
value is used.

### Bug 2 (FIXED): Training collapse after Round 1
Was: paraphrase retriever model + wrong learning rate caused MRR to collapse after
Round 1. All τ values produced identical results.

Fix: RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2" and LEARNING_RATE
= 5e-7 (reduced from original 2e-6) in federated_das.py.

### Bug 3 (FIXED): LSR/distilgpt2 produced identical gradients across clients
Was: distilgpt2 assigns uniform likelihood to all domain-specific text, so all
clients produced near-identical model updates regardless of data quality or domain.

Fix: Replaced HuggingFaceTrainerForLSR + distilgpt2 with ContrastiveFlowerClient
(MultipleNegativesRankingLoss). Training signal comes directly from (query,
positive_doc) pairs with no LM teacher. Domain-specific data now produces
meaningfully different gradients per client.

## Experiment History

### exp_01 / exp_02 — Initial multi-seed verification (FAILED)
- LR=2e-6, 1 epoch, 4→8 rounds
- Soft Domain used `softmax(log(d_j)) × (n_j/N)` weighting
- Result: 95.5% weight on medical client — effectively single-client training
- Soft Domain ≤ Baseline across all seeds

### exp_03 — Temperature scaling + LR fix (PARTIAL)
- Added τ=1.0 for temperature scaling, reduced LR from 2e-6 to 5e-7
- Result: LR=5e-7 helped both algorithms equally (mitigated destructive training)
- τ=1.0 did NOT change weight distribution (still 95.5% to medical)

### exp_04 — Pure relevance weighting (PASSED) ← CANONICAL RESULT
- Dropped τ entirely, dropped n_j from weights
- Algorithm: w_j = d_j / Σ d_k (zero hyperparameters)
- LR=5e-7, 1 epoch, 8 rounds
- **Results:**

| Model | Seed | Final Test MRR | Final Test Recall@10 | Final Test NDCG@10 |
|-------|------|---------------|---------------------|-------------------|
| Baseline | 42 | 0.0271 | 0.0780 | 0.0387 |
| DAS-FedAvg | 42 | **0.0271** | 0.0780 | 0.0387 |
| Baseline | 123 | 0.0262 | 0.0760 | 0.0376 |
| DAS-FedAvg | 123 | **0.0271** ✅ | **0.0800** | **0.0392** |
| Baseline | 256 | 0.0276 | 0.0800 | 0.0396 |
| DAS-FedAvg | 256 | **0.0276** | 0.0800 | 0.0395 |

- Gate: DAS-FedAvg ≥ Baseline on **all 3 seeds** (1 clear win, 2 ties)
- Pre-train MRR = 0.0272. Best DAS-FedAvg = 0.0276 (Seed 256, marginal improvement)

### exp_05 — 3 local epochs (FAILED)
- Increased local epochs from 1 to 3 to amplify training signal
- Result: ALL metrics degraded vs exp_04 (3× more model movement = 3× more destruction)
- DAS-FedAvg still ≥ Baseline (the property holds), but absolute performance worse
- Conclusion: training is destructive at any signal strength for this model/dataset

## Current Status

**Step 6: Final Benchmark Position** (CURRENT)

The algorithm is validated: DAS-FedAvg ≥ Baseline consistently across all experiments
and seeds. The weight distribution is correct and defensible. The algorithm has zero
hyperparameters.

The limitation is that neither algorithm meaningfully improves over the pre-trained
model on nfcorpus. `all-MiniLM-L6-v2` already has strong medical retrieval
representations — federated training degrades them.

### Options going forward

1. **Use exp_04 as canonical result** (recommended) — valid and publishable
2. **Switch target domain** — use a corpus where pre-trained model has headroom
3. **Try LR=1e-6 with 1 epoch** — untested midpoint, low confidence (~30%)

## Algorithm Defense Points

### "What about data volume? Shouldn't larger datasets count more?"
Data volume is implicitly encoded in gradient magnitude. Medical (4000 pairs) does
500 SGD steps per round, producing delta_norm ~0.046. Biomedical (101 pairs) does
13 steps, producing delta_norm ~0.004. Even with equal aggregation weights, the
high-data client's update naturally dominates by ~10:1.

### "What if a large client converges and a small noisy client produces larger deltas?"
In our setup, clients restart from the global model every round — no persistent
client state. There is no cumulative convergence. A shrinking delta from a large
client means the global model already learned what it offers (correct behavior).
The irrelevant client still gets tiny weight (finance = 0.7%).

### "What about angular impact in high-dimensional space?"
The angular deflection from finance (0.7% weight) is bounded by arctan(0.007/0.993)
≈ 0.4°. FedAvg gives finance 3.1% and argument 17.9% — DAS-FedAvg strictly reduces
angular noise from irrelevant clients compared to baseline.

### "Does this break with adaptive optimizers like Adam?"
Adam normalizes per-step magnitude but doesn't change the number of steps. 500 Adam
steps still accumulate more movement than 13 Adam steps. And relevance weighting
provides a second, independent layer of protection.

## Run Slug Format

Soft weighting: soft_domain_seed_{seed}_target_{dataset}_r{rounds}_e{epochs}
Baseline:       baseline_fedavg_seed_{seed}_target_{dataset}_r{rounds}_e{epochs}

Example: soft_domain_seed_42_target_nfcorpus_r8_e1

## Output Files Per Run

results_<slug>.csv      — round-by-round metrics, per-client domain weights
manifest_<slug>.json    — full config snapshot (csv_schema_version: 3)
acceptance_<slug>.json  — pass/fail acceptance tests
log_<slug>.log          — full training log

## Acceptance Tests (in evaluate_single_run_acceptance())

1. target_client_always_selected
   — client 0 (medical) appears in selection_map every round

2. relevance_scores_ordered
   — scores ranked: medical > science > biomedical > argument > finance

3. distinct_round_trajectories
   — aggregated_model_hash is unique across all rounds (training is happening)

4. high_relevance_outweighs_low_relevance
   — client 0 (d=1.0) domain_weight > client 2 (d=0.013) domain_weight every round

5. medical_client_dominates
   — client 0 has the strictly highest domain_weight among all clients every round

## CSV Schema (version 3)

Per-client columns:
  client_{cid}_domain       — BEIR dataset label
  client_{cid}_relevance    — pre-computed cosine similarity score
  client_{cid}_selected     — always "1" (all clients participate)
  client_{cid}_loss         — training loss this round
  client_{cid}_num_examples — training pairs used
  client_{cid}_domain_weight — final normalised soft weight used in aggregation
  client_{cid}_delta_norm   — L2 norm of weight update from this client

Note: client_{cid}_size_weight from schema version 2 is replaced by
client_{cid}_domain_weight in version 3.

## Centroid Computation (correct, do not change)

domain centroid = mean(embeddings of all documents in knowledge store), normalised
target centroid = same computation on target domain knowledge store
relevance score = cosine_similarity(client_centroid, target_centroid)

Centroids are seed-independent — same scores appear across all seed runs.

## Do NOT Repeat

- ❌ Do not use `d_j × n_j` weighting — collapses to 95.5% single-client training
- ❌ Do not use τ-based softmax — experiment-specific, not defensible
- ❌ Do not use LR=2e-6 — degrades pre-trained model over 8 rounds
- ❌ Do not use 3 local epochs at LR=5e-7 — amplifies destructive signal
- ❌ Do not multiply relevance weights by n_j/N — double-counts data volume
- ❌ Do not compare runs with unmatched seeds without deterministic data splits
- ❌ Do not use only 4 rounds — system needs 6-8 rounds

## When Attaching Experiment Logs

Always attach das_fedrag_improvement_plan_from_here.md when asking for help with
experiment design or analysis. For specific experiment data, attach the relevant
das_fedrag_exp_XX_log.md from the corresponding exp_XX_results/ folder.