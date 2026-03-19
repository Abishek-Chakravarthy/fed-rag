# `federated_noisy_qa.py` Walkthrough

## Purpose
This is the main experiment script for the quality-aware federated RAG benchmark.

It is responsible for:

- setting up the dataset
- splitting client data
- injecting controlled noise into one client
- constructing Flower clients
- running the QA-FedAvg simulation
- saving canonical CSV, manifest, and acceptance-report outputs

This is the most important file in the folder if you want the full experiment context.

## What This Script Now Supports
Compared to the earlier version, it now supports:

- multiple seeds
- multiple corruption modes
- configurable local epochs
- configurable federated rounds
- post-aggregation retrieval evaluation
- explicit loss provenance tracking
- acceptance-report generation

## Main Global Configuration
Important constants include:

- `NUM_ROUNDS`
- `NUM_CLIENTS`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `DATASET_NAME`
- `MAX_TRAIN`
- `MAX_EVAL`
- `MAX_DOCS`
- `NOISE_RATIO`
- `NOISE_MODE`
- `NOISY_CLIENT_ID`

There are also run-time globals used by the client factory:

- `CURRENT_SEED`
- `CURRENT_LOCAL_EPOCHS`
- `CURRENT_NOISE_MODE`
- `NOISE_CONTEXT`

Those exist because Flower recreates clients through `client_fn(...)`, so the current run configuration has to be reachable globally.

## Data Split and Noise Injection

### `split_iid(train_pairs, num_clients)`
Creates the base equal split across clients using the current seed.

### `split_noisy(...)`
Starts from the IID split, then corrupts the noisy client.

Supported corruption modes:

- `shuffle`
  Reassign relevant responses inside the noisy client. The current implementation uses a deranged shuffle so the same query should not keep the same response.

- `cross_domain`
  Replace responses using a pool from another dataset such as `SciFact`. This is a stronger corruption mode because it injects truly off-domain supervision.

- `random_negative`
  Replace responses with random document texts from the corpus.

- `mixed`
  Mixes shuffled positives, cross-domain replacements, and random negatives.

The point of these modes is to test whether the QA signal becomes strong enough to produce a measurable alpha trend.

## Helper Utilities

### `build_run_slug(...)`
Creates a unique run identifier used for:

- result CSV path
- manifest path
- acceptance-report path
- log path

This is important because multi-seed and multi-noise-mode runs should never overwrite each other.

### `set_retriever_weights(...)`
Loads aggregated Flower parameters into a fresh retriever.

This is used for post-aggregation evaluation.

### `make_post_aggregation_evaluator()`
Builds the callback passed into `QualityAwareFedAvg`.

After every round, this callback:

1. creates a fresh retriever
2. loads the aggregated weights
3. evaluates the retriever on held-out target-domain queries

### `build_cross_domain_response_pool(...)`
Builds a response pool from another BEIR dataset so cross-domain noise is truly off-domain.

### `sample_random_negative_responses(...)`
Builds same-corpus negative responses that are not the original positive text.

### `deranged_shuffle(...)`
Ensures the shuffle corruption mode is actually corrupting examples instead of leaving some pairs unchanged.

## Manifest and Acceptance Logic

### `write_run_manifest(...)`
Writes metadata about the run, including:

- file locations
- alpha
- seed
- noise mode
- noise ratio
- local epochs
- rounds
- split hashes
- quality-signal provenance

This is the main reproducibility document for a single run.

### `evaluate_single_run_acceptance(...)`
Checks run-level conditions such as:

- `alpha=0` behaves like FedAvg
- higher-loss clients get down-weighted when `alpha>0`
- different rounds produce distinct global model hashes

This is not the cross-alpha benchmark verdict. It is the single-run sanity check.

## Client Construction

### `client_fn(cid)`
This creates a Flower client for one participant.

The main steps are:

1. create retriever and generator
2. create the `RAGSystem`
3. build the client’s local training dataset
4. create `SentenceTransformerTrainingArguments`
5. build `HuggingFaceTrainerForLSR`
6. wrap it in a federated task
7. monkey-patch `fit(...)` so the client reports:
   - `loss`
   - `loss_source`
   - `loss_stage`
   - `noise_mode`

This is how the experiment now makes the loss provenance explicit instead of implicitly trusting the metric key.

## Aggregated Loss Function

### `weighted_average(metrics)`
This still computes the average training loss across clients for the round.

The richer per-client diagnostics now come from the custom strategy rather than from this helper, because Flower does not pass client IDs into this callback.

## Main Entry Point

### `main(...)`
This is the full experiment pipeline.

It does the following:

1. set current run configuration
2. create file paths from `build_run_slug(...)`
3. load the dataset bundle from `prepare_beir_data.py`
4. prepare any cross-domain corruption pool if needed
5. split and corrupt client data
6. run pre-training retrieval evaluation
7. create initial parameters
8. construct `QualityAwareFedAvg`
9. run the Flower simulation
10. write the round-by-round CSV
11. write the single-run acceptance JSON
12. write the run manifest

## Output Files
Each run produces:

- `results_<run_slug>.csv`
- `manifest_<run_slug>.json`
- `acceptance_<run_slug>.json`
- `log_<run_slug>.log`

The CSV now includes both:

- optimization metrics such as `avg_loss`
- retrieval metrics such as `post_mrr`, `post_recall_at_k`, `post_ndcg_at_k`

## How This File Connects to the Others
- uses `prepare_beir_data.py` for dataset setup and retrieval evaluation
- uses `quality_aware_fedavg.py` for server-side aggregation
- is called repeatedly by `run_alpha_sweep.py`

## Best Way to Read This File
Read it in this order:

1. constants and globals
2. `split_noisy(...)`
3. helper functions for evaluation and manifests
4. `client_fn(...)`
5. `main(...)`

That order makes the control flow much easier to follow.
