# QA-FedAvg Agent Context

## What This Project Is

A final year B.Tech project implementing Quality-Aware Federated Averaging (QA-FedAvg)
for Retrieval-Augmented Generation (RAG) systems. The core idea: standard FedAvg
weights client model updates purely by dataset size, ignoring data quality. A client
with a large noisy dataset gets high influence and degrades the global model. QA-FedAvg
introduces a quality signal to downweight noisy clients automatically.

## Repository

Branch: `q-fedrag2` on `https://github.com/Abishek-Chakravarthy/fed-rag`
All experiment code lives in: `zz_coderuns/quality_aware_fedrag/mechanism_benchmark/`
The fed-rag library is in: `src/fed_rag/`

## Execution Environment

- Kaggle notebooks with dual T4 GPUs
- Each alpha value (0.0, 0.3, 0.7, 1.0) runs in a separate notebook copy in parallel
- Notebook: `single_alpha_kaggle_colab.ipynb`
- The notebook clones the branch, installs deps, runs `federated_noisy_qa.py` as subprocess

## The Algorithm (Current State)

### QA-FedAvg Formula

Standard FedAvg:
  w_global = Σ (n_i / n_total) * w_i

QA-FedAvg:
  weight_i = (1 - α) * size_weight_i + α * quality_score_i
  w_global = Σ weight_i * w_i

Quality score:
  quality_score_i = softmax(-β * loss_i)   where β = 5.0

Current quality loss (rank displacement):
  loss_i = mean(max(rank_after - rank_before, 0)) - 0.5 * mean(max(rank_before - rank_after, 0))

  where ranks are measured on a shared probe set of 400 query-document pairs
  BEFORE and AFTER each client's local LSR training.

α = 0.0 → pure FedAvg
α = 1.0 → pure quality-based weighting

## The Training Setup

- Retriever: sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-dim embeddings)
- Generator: distilgpt2 (frozen, used only as teacher signal)
- Training objective: LSR (LM-Supervised Retrieval) — KL divergence between
  retriever cosine similarity scores and frozen LM utility scores
- Dataset: NFCorpus from BEIR benchmark (medical/nutrition IR)
- Knowledge store: InMemoryKnowledgeStore with up to 8000 embedded documents
- Only the query encoder is trained; context encoder stays frozen
- lr = 2e-6, batch_size = 8, 1 epoch per round, 4 rounds, 5 clients

## Mechanism Benchmark Config (what matters for this contribution)

MECHANISM_CLIENT_NOISE_MAP = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.5, "4": 0.9}
MECHANISM_NOISE_MODE = "random_negative"   ← current value after bugfix
MECHANISM_CLIENT_SPLIT_MODE = "equal"
MECHANISM_RETRIEVER = "sentence-transformers/all-MiniLM-L6-v2"

Clients 0, 1, 2 are clean. Client 3 has 50% corrupted data. Client 4 has 90% corrupted.
Corruption = replacing the correct response with a random document from the corpus.

## Key Files

- federated_noisy_qa.py   — main experiment script (1600 lines)
  - Lines 64-76: MECHANISM_* constants
  - Lines 264-314: split_noisy() with noise dispatch (recently fixed)
  - Lines 1100-1130: resolve_benchmark_config()
  - Lines 1133-1240: main()
  - client_fn() and audited_fit(): where quality signal is computed
- prepare_beir_data.py    — data loading, retriever creation, evaluation metrics
- lsr.py                  — LSR trainer with DataParallel fix (in src/fed_rag/)
- single_alpha_kaggle_colab.ipynb — Kaggle execution notebook
- quality_aware_fedavg.py — QualityAwareFedAvg Flower strategy (server-side aggregation)

## Critical Bug History

### Bug 1 (FIXED): Mechanism path hardcoded to shuffle noise
Location: split_noisy() lines 264-290
What happened: The mechanism benchmark path always called deranged_shuffle()
regardless of CURRENT_NOISE_MODE. MECHANISM_NOISE_MODE was logged but never
dispatched. Fixed by adding full noise mode dispatch in the mechanism path.
Evidence: exp_02, exp_03, exp_04a all produced identical model hashes (5706c0d7...)
because shuffle noise is invisible to LSR training.

### Bug 2 (FIXED): DataParallel unwrapping in LSR trainer
Location: lsr.py compute_loss()
What happened: On dual-GPU Kaggle, PyTorch wraps model in DataParallel.
raw_model = model.module if hasattr(model, "module") else model
Fixed in lsr.py.

### Bug 3 (FIXED): Empty optimizer in LSRSentenceTransformerTrainer
Location: lsr.py create_optimizer()
What happened: Parent SentenceTransformerTrainer builds optimizer from loss
module parameters. LSRLoss has no trainable params, so optimizer was empty.
Fixed by overriding create_optimizer() to use model.named_parameters().

## Experiment History

| Exp | Noise | Result | Key Finding |
|-----|-------|--------|-------------|
| Diagnostic | None (single client) | all-MiniLM-L6-v2 +10.3% MRR | Retriever choice critical |
| exp_02 | shuffle 0.3/0.7 | α>0 peaks 1 round later | Quality signal doesn't discriminate |
| exp_03 | shuffle 0.5/0.9 | Identical hashes to exp_02 | Shuffle invisible to LSR |
| exp_04a | random_neg (intended) | Identical hashes — INVALID | Code bug: always ran shuffle |
| exp_04b | random_neg (bugfixed) | NOT YET RUN | Pending |

## Known Constraints (DO NOT REPEAT)

- paraphrase-MiniLM-L3-v2 — degrades under LSR
- lr = 5e-7 — negligible movement
- lr = 5e-6 — training collapse
- epochs >= 2 — training collapse
- rounds >= 6 — model degrades past pre-train baseline
- shuffle noise at any ratio — invisible to LSR, always identical models
- mechanism benchmark without noise dispatch fix — always ran shuffle

## Acceptance Tests (in evaluate_single_run_acceptance())

1. alpha_zero_matches_fedavg — at α=0, all weights equal size weights
2. higher_loss_clients_downweighted — noisiest client weight < its size weight
3. distinct_round_trajectories — all 4 round hashes are different
4. lower_quality_clients_downweighted — all noisy clients (3,4) have lower
   combined weight than all clean clients (0,1,2) — PRIMARY GATE

## Primary Sanity Check for exp_04b

α=0.0 R1 model hash MUST differ from 5706c0d7...
If it matches, random_negative noise is also invisible to LSR (deeper problem).
If it differs, noise is affecting training and the mechanism can be tested.

## Planned Algorithm Change (next after exp_04b)

Replace rank displacement quality signal with delta MRR:
  loss_i = MRR_before - MRR_after   (computed on shared probe set)
  quality_score_i = softmax(β * (-loss_i)) = softmax(β * delta_MRR_i)

Motivation:
- No arbitrary asymmetry coefficient (the current 0.5 has no principled derivation)
- Direct alignment between quality signal and evaluation metric
- Easier to explain and defend
- Server computes it from submitted weights on shared set (more tamper-resistant)

## Output Files Per Run

results_<slug>.csv      — round-by-round metrics, per-client weights and scores
manifest_<slug>.json    — full config snapshot, data split hashes, pre/post metrics
acceptance_<slug>.json  — pass/fail for all four acceptance tests
log_<slug>.log          — full training log

Run slug format:
alpha_<α>_track_<benchmark_mode>_seed_<seed>_mode_<noise_mode>_ratio_<ratio>_r<rounds>_e<epochs>

## When Attaching Experiment Logs

Always attach the most recent experiment_log.md when asking for help.
The log contains model hashes, data hashes, and acceptance results that are
the primary evidence for whether an experiment worked or failed.
