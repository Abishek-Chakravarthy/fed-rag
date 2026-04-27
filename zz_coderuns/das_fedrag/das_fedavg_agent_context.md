# DAS-FedAvg Agent Context

## What This Contribution Is

Domain-Aware Soft Federated Averaging (DAS-FedAvg) for federated RAG systems.

The core idea: in a multi-domain federation (clients holding medical, science,
finance, argument, and biomedical data), training a retriever for a specific target
domain (e.g. medical) is harmed by updates from semantically irrelevant clients
(e.g. finance). DAS-FedAvg computes a domain relevance score for each client as the
cosine similarity between that client's document embedding centroid and the target
domain's centroid, then scales each client's aggregation weight by that score.
Irrelevant clients naturally receive near-zero weight without explicit exclusion.

This is the second of two contributions in the same final year project. The first
contribution (QA-FedAvg) is in a separate codebase and is independent.

## Repository

Branch: `das-fedrag` on `https://github.com/Abishek-Chakravarthy/fed-rag`
All experiment code lives in: `zz_coderuns/das_fedrag/`

## The Algorithm (Current Implementation — Soft Weighting)

Domain relevance score for client j:
  d_j = cos(centroid_j, centroid_target)

  where centroid_j = mean embedding of client j's knowledge store documents,
  normalised to unit length

Aggregation (soft domain weighting — ALL clients participate every round):
  weight_j = (d_j * n_j) / Σ_k (d_k * n_k)

  where n_j = number of training examples reported by client j

There is no threshold τ and no client exclusion. An irrelevant client (d_j ≈ 0)
naturally receives near-zero weight. self.tau is retained in DomainAwareFedAvg
as a stored attribute for logging purposes only — it does not affect selection
or weighting.

Expected weight distribution with equal dataset sizes and the known relevance scores:
  Medical    (d=1.0000): ~53.9%
  Science    (d=0.4791): ~25.8%
  Biomedical (d=0.3212): ~17.3%
  Argument   (d=0.0422):  ~2.3%
  Finance    (d=0.0130):  ~0.7%

## Training Setup

- Retriever: sentence-transformers/all-MiniLM-L6-v2
- Training objective: MultipleNegativesRankingLoss (contrastive, via ContrastiveFlowerClient)
- No generator — LSR/distilgpt2 removed entirely
- Target dataset: NFCorpus (medical/nutrition)
- Other client datasets: SciFact (science), FiQA (finance), ArguAna (argument),
  TREC-COVID (biomedical)
- lr = 2e-6, batch_size = 8, 1 epoch per round, 4 rounds
- 5 clients total, all participate every round

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
- das_experiment_log.md        — attach when asking for help with experiment results

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
= 2e-6 in prepare_multi_domain_data.py and federated_das.py respectively.
Training objective also switched from LSR to contrastive (see Bug 3).

### Bug 3 (FIXED): LSR/distilgpt2 produced identical gradients across clients
Was: distilgpt2 assigns uniform likelihood to all domain-specific text, so all
clients produced near-identical model updates regardless of data quality or domain.

Fix: Replaced HuggingFaceTrainerForLSR + distilgpt2 with ContrastiveFlowerClient
(MultipleNegativesRankingLoss). Training signal comes directly from (query,
positive_doc) pairs with no LM teacher. Domain-specific data now produces
meaningfully different gradients per client.

## Experiment Plan (no experiments run yet)

### Primary runs — seed stability sweep

Run the same soft-domain experiment across 4 seeds in parallel Kaggle notebooks:

| Notebook | Seed | Purpose |
|----------|------|---------|
| 1 | 42  | Primary result |
| 2 | 123 | Stability check |
| 3 | 456 | Stability check |
| 4 | 789 | Stability check |

Command: python federated_das.py --seed <SEED> --rounds 4 --local-epochs 1 --target nfcorpus

### Ablation — vanilla FedAvg baseline

Run with --baseline flag. This overrides all relevance scores to 1.0, degenerating
soft weighting to pure size-based FedAvg with no domain scoring. Use seed=42.

Command: python federated_das.py --seed 42 --rounds 4 --local-epochs 1 --target nfcorpus --baseline

Key comparison: soft_domain_seed_42 MRR vs baseline_fedavg_seed_42 MRR.
This directly proves the contribution: domain-aware weighting improves over
naive aggregation.

## Run Slug Format

Soft weighting: soft_domain_seed_{seed}_target_{dataset}_r{rounds}_e{epochs}
Baseline:       baseline_fedavg_seed_{seed}_target_{dataset}_r{rounds}_e{epochs}

Example: soft_domain_seed_42_target_nfcorpus_r4_e1

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

## When Attaching Experiment Logs

Always attach das_experiment_log.md when asking for help with experiment results.
The log contains model hashes, per-round weight distributions, and acceptance results.