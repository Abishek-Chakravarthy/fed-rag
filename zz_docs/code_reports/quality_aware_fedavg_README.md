# `quality_aware_fedavg.py` Walkthrough

## Purpose
This file defines the custom Flower strategy `QualityAwareFedAvg`.

It is the server-side logic that replaces plain FedAvg with:

- size-aware weighting
- inverse-loss quality weighting
- optional post-aggregation evaluation
- round-by-round audit logging

If `federated_noisy_qa.py` is the experiment driver, this file is the aggregation brain.

## What the Strategy Is Solving
Standard FedAvg only uses client dataset size. This strategy changes the aggregation rule so that clients with lower training loss get more influence.

The implemented weight is:

- size weight from `num_examples`
- quality score from inverse loss
- blended by `alpha`

So:

- `alpha = 0.0` -> standard FedAvg
- `alpha > 0.0` -> quality-aware aggregation

## Main Class

### `class QualityAwareFedAvg(FedAvg)`
This subclasses Flower’s `FedAvg` and overrides `aggregate_fit(...)`.

## Constructor
The important inputs are:

- `alpha`: controls how strongly quality affects weighting
- `epsilon`: prevents division by zero in inverse-loss scoring
- `post_aggregation_evaluator`: optional callback to evaluate the global model after each round

The constructor also stores:

- `round_quality_info`: the full per-round audit trail
- `last_global_ndarrays`: the previous global model, used to measure update magnitudes

## Helper Methods

### `_client_sort_key(cid)`
Flower result order is not something we should trust for stable reporting.

This helper sorts client IDs so that:

- numeric IDs are ordered numerically
- non-numeric IDs still work safely

This is what fixes the attribution/order problem in the experiment logs and CSVs.

### `_hash_ndarrays(ndarrays)`
Produces a deterministic MD5 hash of a model snapshot.

This is useful for proving that different rounds or different alphas really produced different global models.

## Core Method

### `aggregate_fit(server_round, results, failures)`
This is the central method in the file.

It does the following in order:

1. Read each client’s:
   - model weights
   - `num_examples`
   - reported `loss`
   - extra metric provenance such as `loss_source`
2. Sort clients by `cid`.
3. Compute inverse-loss quality scores.
4. Compute dataset-size weights.
5. Blend both using `alpha`.
6. Aggregate model layers using the blended weights.
7. Measure client update norms and global update norm.
8. Optionally run post-aggregation evaluation.
9. Save a detailed record into `round_quality_info`.

## What Gets Logged Per Client
Each client record contains:

- `cid`
- `loss`
- `num_examples`
- `quality_score`
- `size_weight`
- `combined_weight`
- `delta_norm`
- `loss_source`
- `loss_stage`

This makes the CSVs and acceptance checks much more trustworthy than a plain list of per-round losses.

## What Gets Logged Per Round
Each round record contains:

- `round`
- `alpha`
- `aggregated_loss`
- `aggregated_delta_norm`
- `aggregated_model_hash`
- `post_eval_metrics`
- `client_records`

So this object is the main evidence source for:

- whether alpha changes the weights
- whether alpha changes the model trajectory
- whether retrieval metrics improve after aggregation

## Post-Aggregation Evaluation
The strategy can call a callback after it computes the new global model.

That callback returns retrieval metrics such as:

- `mrr`
- `recall_at_k`
- `ndcg_at_k`

This is how the experiment now checks whether QA-FedAvg improves actual retrieval, not just training loss.

## How This File Connects to the Others
- `federated_noisy_qa.py` creates the strategy and passes:
  - `alpha`
  - `initial_parameters`
  - `fit_metrics_aggregation_fn`
  - `post_aggregation_evaluator`
- `run_alpha_sweep.py` indirectly depends on this file because every benchmark run uses this strategy.

## Reading Tips
When reading side by side with the code, focus on:

1. How `losses` become `quality_scores`.
2. How `quality_scores` combine with `size_weights`.
3. How client IDs are stabilized before reporting.
4. How `post_eval_metrics` are attached to the same round record as the aggregation weights.

That is the shortest path to understanding the whole QA-FedAvg implementation.
