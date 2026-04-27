### Files to give the new agent (priority order)

| # | File | What it provides |
|---|------|-----------------|
| 1 | `mechanism_benchmark/context.md` | **Start here** — project overview, QA-FedAvg formula, architecture, experiment history, bugs, current state |
| 2 | `mechanism_benchmark/improvement_plan_from_here.md` | Master plan with decision tree, gates, and "Do NOT Repeat" list |
| 3 | `mechanism_benchmark/federated_noisy_qa.py` | Main script (1600 lines) — the noise dispatch fix, config resolution, training loop |
| 4 | `mechanism_benchmark/single_alpha_kaggle_colab.ipynb` | Kaggle notebook — needs `NOISE_MODE` changed to `"random_negative"` |
| 5 | `mechanism_benchmark/exp_03_results/experiment_log.md` | Proof that shuffle noise is invisible to LSR |
| 6 | `mechanism_benchmark/exp_04a_results/experiment_log.md` | Code bug documentation (mechanism path ignored noise mode) |
| 7 | `mechanism_benchmark/lsr_training_check/diagnostic_summary.json` | Retriever diagnostic results |
| 8 | `src/fed_rag/trainers/huggingface/lsr.py` | LSR trainer with DataParallel fix |
| 9 | `mechanism_benchmark/prepare_beir_data.py` | Data loading config |