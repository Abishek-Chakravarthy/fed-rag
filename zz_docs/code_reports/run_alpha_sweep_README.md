# `run_alpha_sweep.py` Walkthrough

## Purpose
This file is the benchmark runner.

It does not train models itself. Instead, it repeatedly launches `federated_noisy_qa.py` with different settings and then summarizes the outputs.

Its job is to answer questions like:

- Does the alpha trend show up consistently across seeds?
- Which corruption mode produces the clearest quality gap?
- Do post-aggregation retrieval metrics improve along with the aggregation weights?
- Does the benchmark support the thesis claim or suggest revising it?

## What It Sweeps Over
The script now supports:

- `alpha`
- `seed`
- `noise_mode`
- `noise_ratio`
- `rounds`
- `local_epochs`

So this file is the experiment matrix controller.

## Default Benchmark Space
By default it is configured for:

- alphas: `0.0, 0.2, 0.4, 0.6, 0.8, 1.0`
- seeds: `42, 52, 62`
- noise modes:
  - `shuffle`
  - `cross_domain`
  - `random_negative`
  - `mixed`
- noise ratio: `0.7`

These defaults are meant to produce statistics, not just one illustrative run.

## Core Helpers

### `build_run_slug(...)`
This mirrors the naming logic in `federated_noisy_qa.py`.

That matters because the sweep runner has to know exactly where each run’s:

- result CSV
- manifest
- acceptance report

will be written.

### `mean_and_std(values)`
Computes mean and population standard deviation for summary tables.

This is the core of the “report mean and variance, not just one run” requirement.

### `ratio_slug(value)`
Converts a float such as `0.70` into a filename-safe slug like `0p70`.

## Running One Experiment

### `run_one(...)`
Launches one subprocess:

- calls `federated_noisy_qa.py`
- passes the selected alpha, seed, noise mode, noise ratio, rounds, and local epochs
- captures stdout into a sweep log
- returns the expected artifact paths and subprocess status

This function is the bridge between the matrix runner and the training script.

## Summarizing a Noise Setting

### `summarize_group(run_records, noise_mode, noise_ratio, rounds, local_epochs)`
This function groups all runs for one corruption setting and summarizes them by alpha.

For each alpha it computes:

- mean/std of final training loss
- mean/std of final post-aggregation `MRR`
- mean/std of final post-aggregation `Recall@k`
- mean/std of final post-aggregation `NDCG@k`
- mean/std of the noisy client’s final aggregation weight
- mean/std of final global update norm

It also aggregates acceptance signals such as:

- `alpha=0` FedAvg parity
- higher-loss down-weighting for `alpha>0`
- presence of all reproducibility artifacts
- whether the best alpha is an interior value, which is the simple operational test for an inverted-U pattern

## Output Artifacts
For each `(noise_mode, noise_ratio)` group, the runner writes:

- `summary_mode_<...>.csv`
- `acceptance_summary_mode_<...>.json`

These are the documents you should look at when deciding:

- which corruption mode is strongest
- whether alpha trends are stable across seeds
- whether the thesis claim is empirically supported

## Main Function

### `main()`
The high-level flow is:

1. parse CLI arguments
2. loop over noise modes and noise ratios
3. for each group, loop over all alphas and seeds
4. launch a run for each combination
5. summarize the whole group
6. print the best-alpha evidence and compact statistics

## How This File Connects to the Others
- launches `federated_noisy_qa.py`
- depends on the artifact naming convention used there
- indirectly exercises `quality_aware_fedavg.py`
- indirectly relies on `prepare_beir_data.py` through the training script

## What to Watch For While Reading
If you are reading this file side by side with the code, focus on:

1. how a single run is launched
2. how artifact paths are reconstructed
3. how final-round statistics are aggregated across seeds
4. how acceptance criteria are turned into benchmark-level summaries

## Practical Interpretation
This file is the answer to the question:

“After making the experiment reproducible and better instrumented, how do we tell whether QA-FedAvg is really working?”

It is the file that converts many individual runs into evidence you can reason about.
