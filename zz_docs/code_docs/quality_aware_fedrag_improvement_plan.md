# Improvement Plan: `quality_aware_fedrag`

This document is the canonical improvement roadmap for the QA-FedAvg federated RAG experiments under `fed-rag/zz_coderuns/quality_aware_fedrag/`. It supersedes ad-hoc notes by tying each change to concrete files, success criteria, and known failure modes observed in recent runs.

**Primary code paths**

| File | Role |
|------|------|
| `federated_noisy_qa.py` | End-to-end FL simulation, client `audited_fit`, data splits, noise injection, CSV/manifest |
| `quality_aware_fedavg.py` | `QualityAwareFedAvg` strategy: softmax quality scores, aggregation, best-round tracking |
| `prepare_beir_data.py` | BEIR loading, knowledge store, `shared_quality` / `server_val` / `final_test` splits |
| `run_alpha_sweep.py` | Multi-alpha, multi-seed sweep; maps `shuffle` → equal split, `hard_negative` → unequal |
| `compile_alpha_results.py` | Aggregates `results_*.csv`, plots alpha curves |

---

## 1. Current State (Baseline)

### 1.1 What already works

- **Mechanical correctness**: FedAvg at `alpha=0` matches size-only weights; distinct model hashes per round; manifests and acceptance JSONs are produced.
- **Instrumentation**: Per-client shared-quality diagnostics, server validation, final test, best-round selection by `server_val_mrr` then `ndcg_at_k`.
- **Two official benchmarks**: Control (`shuffle`, equal split, noisy client 4) vs stress (`hard_negative`, unequal split, 40% data on noisy client).

### 1.2 Known issues (must drive the plan)

1. **Round-1 quality inversion (stress benchmark)**  
   The noisy client often has the *lowest* `shared_rank_displacement` loss in round 1 because it has more data and produces larger rank moves before corruption dominates. At high `alpha`, aggregation can assign **most weight to the noisy client** in round 1, which poisons later rounds.

2. **Weak end-to-end retrieval lift**  
   Absolute MRR stays very low (~0.02) on a large corpus; small changes in alpha are often within metric noise unless you aggregate seeds and fix evaluation design.

3. **Thesis vs code mismatch**  
   Report text may still describe inverse-LSR-loss weighting and small-scale settings; the running code uses **shared rank displacement** + **softmax(-β·z-score(losses))** for quality scores. Any publication must describe the implemented pipeline.

4. **Best-round selection can mask late-round behavior**  
   If the best round is always early (e.g. round 2), improvements that appear in rounds 3–4 are invisible in `final_test_*`. The plan should either warm up quality weighting or report both “best val” and “last round” metrics where useful.

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Make **QA-FedAvg beat FedAvg (`alpha=0`)** on **final test MRR / NDCG@k** in at least the **control** benchmark, with **multi-seed** stability.
- In the **stress** benchmark, **avoid harmful round-1 aggregation** when the noisy client is oversized.
- Keep the implementation **auditable**: same CSV schema versioning, manifests, acceptance checks.

### 2.2 Non-Goals (for this phase)

- Replacing the entire FL backend or retriever family (stay on MiniLM + distilgpt2 unless a separate experiment branch is opened).
- Full DAS-FedAvg (domain selection); that lives in a separate code path (`das_fedrag`).

---

## 3. Phase A — Fix the Quality Signal (Highest Priority)

### A.1 Per-example normalization of the client quality loss

**Problem**: Raw rank displacement sums scale with “how much the model moved,” which correlates with **dataset size** and **update norm**, not only with “is this client harmful.”

**Change**

- In `federated_noisy_qa.py`, when computing the scalar passed as `metrics["loss"]` to Flower, divide the aggregated rank-based loss by `max(1, num_train_examples)` or by `sqrt(n_j)` (pick one and document it).
- Alternatively, normalize by the L2 norm of the client weight delta (if available without heavy cost) so “big client” does not automatically look best.

**Files**: `federated_noisy_qa.py` (`summarize_rank_displacement`, `audited_fit`), optionally `quality_aware_fedavg.py` if you pass `n_examples` into a custom metric before z-scoring.

**Acceptance**

- In **round 1** of the stress run, the noisy client should **not** receive the highest quality score purely because of mass of data (compare to clean clients with equal hypothetical loss per example).

### A.2 Warm-up schedule for `alpha` (mandatory for stress)

**Problem**: Quality scores are unreliable before local training has interacted with corrupted labels long enough.

**Change**

- Add strategy or driver logic: e.g. `alpha_effective = 0` for `server_round <= warmup_rounds`, then `alpha_effective = alpha` after.
- Implement in **one** place only: either `QualityAwareFedAvg.aggregate_fit` (knows round index) or a thin wrapper in `federated_noisy_qa.py` that passes a round-dependent alpha via config (cleaner: strategy field `warmup_rounds`).

**Files**: `quality_aware_fedavg.py`, `federated_noisy_qa.py` (strategy construction).

**Suggested default**: `warmup_rounds=1` or `2` for stress; `0` or `1` for control.

**Acceptance**

- Stress benchmark: `client_4_weight` at round 1 should not exceed FedAvg size weight when `alpha` is high unless ablation explicitly disables warm-up.

### A.3 Optional: blend training loss with rank displacement

**Problem**: LSR training loss reacts quickly to wrong query–response pairs; rank displacement on shared probes is more semantically aligned with retrieval but lags.

**Change**

- Define `loss = λ * rank_displacement_loss + (1-λ) * train_loss` (both client-normalized), with `λ` in config (e.g. 0.7).
- Log both components in CSV for analysis.

**Files**: `federated_noisy_qa.py` (`audited_fit`).

**Acceptance**

- Noisy client’s **train_loss** should already be higher than clean clients in early rounds on shuffle/hard-negative; verify in CSV before relying on this for weighting.

### A.4 Tuning `quality_beta` and stability

**Problem**: Softmax sharpness (`quality_beta`) changes how aggressively the server discounts outliers.

**Change**

- Log `quality_beta` in manifest (already partially there).
- Run a small grid: e.g. `β ∈ {3, 5, 10}` on control + one seed before scaling.

**Files**: `quality_aware_fedavg.py`, `federated_noisy_qa.py` (constants / CLI).

---

## 4. Phase B — Evaluation and Reporting

### B.1 Multi-seed runs as default for “final” tables

**Problem**: Single-seed (e.g. 42) is insufficient for claims about alpha.

**Change**

- Use `run_alpha_sweep.py` with `--seeds 42 52 62` for final tables.
- Summarize mean ± std in `compile_alpha_results.py` output (already partially supported).

**Files**: `run_alpha_sweep.py`, `compile_alpha_results.py`.

### B.2 Report both selection policies

**Problem**: “Best validation round” hides collapse after the peak.

**Change**

- Add optional columns or a second CSV section: `final_test_mrr_last_round`, `server_val_mrr_last_round` (or compute in compile script from per-round rows).
- In the thesis, show **best-val** as primary and **last-round** as diagnostic.

**Files**: `federated_noisy_qa.py` (writer), `compile_alpha_results.py`.

### B.3 Align thesis text with implementation

**Problem**: Methodology chapter may still describe inverse-loss on LSR loss only.

**Change**

- Update notation: quality score = softmax on **z-scores of client-reported losses**, where losses are **shared rank displacement** (and optional LSR blend).
- Document `epsilon`, `quality_beta`, `warmup_rounds`, and normalization in one “Implementation” subsection.

**Files**: LaTeX under `template/ context/` (outside this repo path if duplicated elsewhere), **no code change required** for experiments.

---

## 5. Phase C — Training Regime (If Metrics Stay Flat)

**Problem**: If retrieval metrics barely move, QA-FedAvg differences will appear as noise.

**Change (incremental, one at a time)**

1. **Increase rounds** to 6–8 after warm-up is validated (do not increase rounds and learning rate at once).
2. **Learning rate sweep** on a narrow band (e.g. `1e-6`–`5e-6`) with fixed rounds.
3. **Avoid** raising local epochs first; it increases client drift and can worsen FL stability.

**Files**: `federated_noisy_qa.py` (`NUM_ROUNDS`, `LEARNING_RATE`), CLI args.

**Acceptance**

- Pre vs post `server_val_mrr` should show a **clear upward trend** for at least FedAvg on control before attributing gains to QA-FedAvg.

---

## 6. Phase D — Benchmark Hygiene

### D.1 Control vs stress parity

- Keep **noise_ratio** semantics explicit in folder names: control uses `shuffle` + `0.70`; stress uses `hard_negative` + `0.80` (as in current `latest_results_*`).
- Document **noisy_client_id** = `"4"` and **unequal** fractions: four clients at 0.15 each, one at 0.40 (verify in `split_unequal_noisy`).

### D.2 Acceptance tests (extend)

Current checks: `alpha_zero_matches_fedavg`, `distinct_round_trajectories`, `higher_loss_clients_downweighted` (max-loss client).

**Add**

- `noisy_client_downweighted_when_applicable`: when `noise_mode == hard_negative` and `server_round > warmup_rounds`, `client_4_weight` < `client_4_size_weight` for at least 50% of rounds (tunable).
- `warmup_respected`: when `warmup_rounds > 0`, first round uses `alpha_effective == 0`.

**Files**: `federated_noisy_qa.py` (`evaluate_single_run_acceptance`).

---

## 7. Suggested Execution Order

1. Implement **A.2 warm-up** + **A.1 normalization** (minimal code, high impact).
2. Re-run **control** sweep (4 alphas × 3 seeds) and compile plots.
3. If control shows **α>0** beating **α=0** on mean final test MRR/NDCG, proceed to **stress** with same settings.
4. If metrics still flat, apply **Phase C** (rounds + small LR sweep) **before** further algorithm changes.
5. Optional **A.3** blend if rank-only signal remains too noisy.
6. Update thesis text (**B.3**) and figures from **compiled** outputs, not hand-copied tables.

---

## 8. Success Criteria (Definition of Done)

| Criterion | Target |
|-----------|--------|
| Control benchmark | Mean `final_test_mrr` (or NDCG@k) at best α **>** FedAvg at α=0 across ≥2 of 3 seeds |
| Stress benchmark | No round-1 inversion (noisy client weight ≤ size weight when α high and warm-up on) OR mean final test ≥ FedAvg without catastrophic early round |
| Robustness | Report mean ± std over seeds; best α not always 0 or always 1 |
| Reproducibility | Manifest + CSV + acceptance JSON for each run; same `build_run_slug` |

---

## 9. Risk Register

| Risk | Mitigation |
|------|------------|
| Shared probes leak information about “good” ranking | Acceptable for a **server-coordinated** diagnostic; document as non-private signal |
| Best-round picks early round always | Report last-round metrics; increase warm-up so “good” rounds exist later |
| Hard negative too easy | Already using top wrong doc; revisit `sample_hard_negative_responses` if needed |
| Runtime | Reduce `MAX_TRAIN` temporarily for debugging; restore full scale for final runs |

---

## 10. References Inside This Repo

- Original informal checklist: `zz_coderuns/quality_aware_fedrag/improvement_plan.md`
- Data pipeline details: `zz_docs/code_docs/prepare_beir_data_report.md` (note: paths/constants may differ between `baseline1` and `quality_aware_fedrag`; always read the **actual** `prepare_beir_data.py` next to the experiment)

---

*Document version: 1.0 — aligned with `quality_aware_fedrag` layout and recent control/stress result folders.*
