# DAS-FedAvg Agent Context

## What This Contribution Is

Domain-Aware Selective Federated Averaging (DAS-FedAvg) for federated RAG systems.

The core idea: in a multi-domain federation (e.g. clients holding medical, science,
finance, argument, and biomedical data), training a retriever for a specific target
domain (e.g. medical) is harmed by updates from semantically irrelevant clients
(e.g. finance). DAS-FedAvg computes a domain relevance score for each client as the
cosine similarity between that client's document embedding centroid and the target
domain's centroid. Clients with low relevance are down-weighted or excluded.

This is the second of two contributions in the same final year project. The first
contribution (QA-FedAvg) is in a separate codebase and is independent.

## Repository

Branch: `das-fedrag` on `https://github.com/Abishek-Chakravarthy/fed-rag`
All experiment code lives in: `zz_coderuns/das_fedrag/`

## The Algorithm (Current State — Hard Threshold)

Domain relevance score for client j:
  d_j = cos(centroid_j, centroid_target)

  where centroid_j = mean embedding of client j's knowledge store documents

Client selection:
  S_t = { j | d_j > τ }   (only clients above threshold participate)

Aggregation (standard FedAvg over selected clients):
  w_global = Σ_{j in S_t} (n_j / n_S) * w_j

  where n_S = Σ_{j in S_t} n_j

## Planned Algorithm Change (MUST implement before any valid experimentation)

Replace hard threshold selection with soft weighting. Every client participates
but their weight is scaled by domain relevance:

  weight_i = normalise( d_i * (n_i / n_total) )

This eliminates the τ hyperparameter entirely. An irrelevant client (d_i ≈ 0)
naturally receives near-zero weight without needing to be explicitly excluded.
τ hard selection becomes a special case (threshold at 0) that can be shown as
an ablation if needed.

Motivation:
- No arbitrary threshold hyperparameter to justify
- Degrades gracefully (finance client in medical federation → weight ≈ 0 naturally)
- Easier to explain: "each client's contribution is scaled by how relevant their
  domain is to the target"
- Eliminates the Round 1 all-clients fallback problem (all clients can participate
  in all rounds, just with very different weights)

## Training Setup

- Retriever: sentence-transformers/all-MiniLM-L6-v2
- Generator: distilgpt2 (frozen)
- Training objective: LSR (same as QA-FedAvg contribution)
- Target dataset: NFCorpus (medical/nutrition)
- Other client datasets: SciFact (science), FiQA (finance), ArguAna (argument),
  TREC-COVID (biomedical)
- lr = 2e-6, batch_size = 8, 1 epoch per round, 4 rounds
- 5 clients total

## Client Domain Relevance Scores (computed, correct)

| Client | Domain | Relevance Score |
|--------|--------|-----------------|
| 0 | Medical (NFCorpus) | 1.0000 ← TARGET |
| 1 | Science (SciFact) | 0.4791 |
| 4 | Biomedical (TREC-COVID) | 0.3212 |
| 3 | Argument (ArguAna) | 0.0422 |
| 2 | Finance (FiQA) | 0.0130 |

These scores are semantically correct and already validated. No changes needed.

## Key Files

- domain_aware_fedavg.py   — DomainAwareFedAvg Flower strategy (server-side)
- federated_das.py         — main experiment script
- prepare_multi_domain_data.py — multi-domain data loading, centroid computation
- single_tau_kaggle_colab.ipynb — Kaggle execution notebook
- das_experiment_log.md    — experiment results (all invalid due to bugs below)

## Critical Bugs (BOTH must be fixed before any valid experimentation)

### Bug 1 (NOT YET FIXED): Round 1 proxy-logical CID mapping
Location: domain_aware_fedavg.py, configure_fit() and aggregate_fit()

What happens: Flower assigns proxy CIDs to clients at simulation startup. The
strategy maps proxy CID → logical CID (0,1,2,3,4) only after seeing the first
aggregate_fit() results (by reading fit_res.metrics["logical_cid"]).

Consequence: In Round 1, proxy_to_logical_cid is empty, so _select_clients()
sees all clients as "unmapped" and selects ALL of them regardless of τ. Since
the model degrades after Round 1 in the current training setup, Round 1 is
always selected as the best round, making τ completely invisible in results.

Evidence: All τ values (0.0, 0.2, 0.4, 0.6) produced identical final metrics.

Fix approach: Pre-register the logical CID mapping at simulation startup.
In Flower's start_simulation(), client_fn receives a context with
context.node_config["partition-id"] which IS deterministic (matches the
partition index). The mapping from partition-id to logical CID is known
before Round 1. Pass this mapping into the strategy at __init__ time so
configure_fit() can apply τ filtering from Round 1 onwards.

Specifically in federated_das.py:
- Build a dict: partition_to_logical = {"0": "0", "1": "1", ...} or
  whatever the partition-id → logical CID mapping is
- Pass it to DomainAwareFedAvg.__init__ as proxy_to_logical_cid
- In domain_aware_fedavg.py __init__, initialise self.proxy_to_logical_cid
  with this pre-registered mapping instead of an empty dict

### Bug 2 (NOT YET FIXED): Training collapse after Round 1
What happens: All clients degrade significantly after Round 1
(Pre-train MRR ~0.0159 → R1 MRR ~0.0112 → R2+ MRR ~0.000-0.001)

Root cause: Same issue solved in QA-FedAvg contribution — the paraphrase
retriever model was used and learning rate was not tuned.

Fix: Use sentence-transformers/all-MiniLM-L6-v2 retriever and lr=2e-6.
This combination is confirmed working in the QA-FedAvg diagnostic (+10.3% MRR).
Update RETRIEVER_MODEL in prepare_multi_domain_data.py accordingly.

## Experiment History

All current results are INVALID due to Bug 1 and Bug 2.

Expected result after fixes:
- τ=0.0 (all clients, weighted by size): baseline
- Soft weighting by d_i: medical + biomedical + science get most weight,
  finance and argument get near-zero weight → better target domain MRR
- Ablation: compare soft weighting vs hard threshold τ=0.4 to show
  soft weighting is more robust

## Centroid Computation (already correct, do not change)

domain centroid = mean(embeddings of all documents in knowledge store), normalised
target centroid = same computation on target domain knowledge store
relevance score = cosine_similarity(client_centroid, target_centroid)

Centroids are cached in .centroid_cache/ to avoid recomputation.

## Output Files Per Run

results_<slug>.csv      — round-by-round metrics, per-client selection and weights
manifest_<slug>.json    — full config snapshot
acceptance_<slug>.json  — pass/fail acceptance tests
log_<slug>.log          — full training log

Run slug format: tau_<τ>_seed_<seed>_target_<dataset>_r<rounds>_e<epochs>

## Acceptance Tests (in evaluate_single_run_acceptance())

1. tau_zero_selects_all — at τ=0, all clients selected every round
2. positive_tau_excludes_some_clients — at τ>0, some clients excluded after R1
3. target_client_always_selected — client 0 (medical) always participates
4. selection_matches_threshold_policy — selected set matches τ policy
5. relevance_scores_ordered — scores correctly ordered by domain relevance
6. distinct_round_trajectories — model hashes differ across rounds

Note: Acceptance tests 2 and 4 currently fail because of Bug 1.
After switching to soft weighting, tests 1 and 2 need to be updated since
there is no longer a hard threshold — replace with:
  "high_relevance_clients_outweigh_low_relevance_clients" — client 0 weight
  > client 2 weight in every round.

## When Attaching Experiment Logs

Always attach das_experiment_log.md when asking for help with this contribution.
The log contains model hashes, per-round selection maps, and acceptance results.

## Priority

Fix Bug 1 (CID mapping) first, then Bug 2 (training collapse), then switch
to soft weighting formulation, then run experiments. Do not run any experiments
until both bugs are fixed — results will be invalid.
