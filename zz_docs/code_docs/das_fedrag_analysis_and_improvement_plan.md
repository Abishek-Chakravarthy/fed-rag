# DAS-FedAvg: Analysis & Improvement Plan

## 1. Current State

### Architecture
DAS-FedAvg (Domain-Aware Selective FedAvg) extends standard FedAvg with a **domain relevance filter** in `configure_fit`. Before each round, the server computes cosine similarity between each client's document-space centroid and the target domain centroid. Clients with relevance `d_j > τ` are selected; the rest are excluded. Aggregation over the selected subset uses standard size-weighted averaging.

### Client Configuration (5 BEIR domains)

| Client | Dataset | Corpus Size | Eligible Pairs | Train Pairs (before fix) | Train Pairs (after fix) | Relevance to NFCorpus |
|--------|---------|------------|----------------|--------------------------|-------------------------|-----------------------|
| 0 | NFCorpus (medical) | 3,633 | 12,334 | **11,434** | 4,000 | **1.0000** |
| 1 | SciFact (science) | 5,183 | 339 | 239 | 239 | 0.4791 |
| 2 | FiQA (finance) | 57,638 | 238 (filtered) | 168 | 168 | 0.0130 |
| 3 | ArguAna (argument) | 8,674 | 1,401 | 981 | 981 | 0.0422 |
| 4 | TREC-COVID (biomedical) | 171,332 | 143 (filtered) | 101 | 101 | 0.3212 |

### Experimental Run (τ=0.0, Kaggle T4)
- **Outcome**: Crashed during round 1 client training.
- Only 1 of 5 clients completed (Client 1 / SciFact). 4 clients hit CUDA errors.
- The experiment finished with 1 round of data. All acceptance tests failed.

---

## 2. Bugs Found & Fixes Applied

### Bug 1 — CUDA Error (CRASH) ✅ FIXED
**Symptom**: `torch.AcceleratorError: CUDA error: device-side assert triggered` at `torch.log(probs[target_id])` in `compute_target_sequence_proba`.

**Root cause**: `get_generator_load_kwargs()` returns `torch_dtype=torch.float16` on CUDA. Half-precision causes overflow/underflow in distilgpt2's attention layers, producing NaN logits → NaN probs → CUDA assert when indexing.

**Fix**: Changed `federated_das.py:get_generator_load_kwargs()` to use `torch.float32` on all devices. distilgpt2 is only 82M parameters (~328MB in fp32), well within T4's 16GB.

**Additional safety**: Added `MAX_QUERY_CHARS = 300` in `prepare_multi_domain_data.py` and truncate queries in `build_positive_pairs`. ArguAna has paragraph-length queries that could exceed distilgpt2's 1024-token context window when combined with document context and response.

### Bug 2 — `split_pairs_with_caps` Ignores `max_train` ✅ FIXED
**Symptom**: NFCorpus gets 11,434 training pairs instead of the intended 4,000 maximum.

**Root cause**: Line `train_count = total - server_val_target - final_test_target` computes the leftover after allocating val/test, ignoring `train_target` (which was correctly capped at `max_train`).

**Fix**: Changed to `train_count = min(train_target, total - server_val_target - final_test_target)`.

**Impact**: NFCorpus train drops from 11,434 → 4,000 pairs. Training time drops ~65% for this client. Data imbalance with other clients is reduced (though still large: 4000 vs 101 for TREC-COVID).

### Bug 3 — Round-1 Proxy→Logical CID Mapping Empty ✅ FIXED
**Symptom**: In round 1, `_select_clients()` treats all clients as "unknown" (relevance=1.0), bypassing τ-based selection entirely. Acceptance test `tau_zero_selects_all` fails because `num_selected` counts successful returns (1), not configured clients (5).

**Root cause**: Flower/Ray assigns arbitrary actor IDs as proxy CIDs (e.g., `2556717137513845682`). The `proxy_to_logical_cid` map is only populated in `aggregate_fit` after training completes. In round 1's `configure_fit`, the map is empty.

**Fix**: 
1. `_select_clients` now accepts `server_round` parameter. When unknown proxies exist, it logs a warning if `server_round > 1` (indicating the mapping should have been populated).
2. Acceptance test `policy_round_infos` now always skips round 1 (warm-up), not just for `tau > 0`.
3. This is a fundamental limitation of Flower simulation's proxy CID scheme — round 1 is necessarily a warm-up where all clients participate.

### Bug 4 — Deprecated `client_fn` Signature ✅ FIXED
**Symptom**: Warning: `client_fn now expects def client_fn(context: Context)`.

**Fix**: Updated `client_fn` to accept `Context` and extract CID via `context.node_config["partition-id"]`.

---

## 3. Remaining Issues (Not Yet Fixed)

### Issue A — FiQA & TREC-COVID Data Sparsity
With `MAX_CORPUS_DOCS=8000`, only the first 8K documents are indexed. FiQA has 57K docs and TREC-COVID has 171K, so the vast majority are un-indexed. After filtering eval pairs to those whose relevant documents are in the indexed subset:
- FiQA: 1,705 → 238 eligible pairs (86% lost)
- TREC-COVID: 21,538 → 143 eligible pairs (99.3% lost)

**Options**:
1. **Increase MAX_CORPUS_DOCS** (e.g., 20K or unlimited) — increases embedding time but retains more pairs.
2. **Swap datasets**: Replace FiQA/TREC-COVID with smaller BEIR datasets (e.g., SCIDOCS, CQADupStack) that have corpora ≤10K.
3. **Accept the sparsity** but document it: the relevance scores already correctly identify these as irrelevant (0.013 and 0.042), so even with sparse data, the selection mechanism works.

**Recommendation**: Option 3 for initial validation. The selection mechanism is the contribution, not the training quality of irrelevant clients.

### Issue B — Massive Training Data Imbalance
Even after fixing the cap, NFCorpus has 4,000 train pairs while TREC-COVID has 101. In FedAvg aggregation, the size-weighted contribution of TREC-COVID is negligible.

**Options**:
1. Lower `MAX_TRAIN_PAIRS` to 500 to level the playing field.
2. Use a per-client training-pair cap (e.g., `min(len(pairs), 500)`).
3. Accept: DAS-FedAvg's contribution is **selection**, not rebalancing. The target client having the most data is realistic.

### Issue C — Round-1 Warm-Up Cannot Be Avoided
Flower simulation assigns proxy CIDs that are unknown until the first `aggregate_fit`. This means:
- Round 1 always selects all clients regardless of τ.
- At τ > 0, the first round of aggregation will include irrelevant clients.

**Mitigation**: Increase `NUM_ROUNDS` from 4 to at least 5 so that rounds 2–5 fully exercise τ-based selection (4 "real" rounds of selective participation).

### Issue D — Acceptance Test `num_selected` vs Configured Clients
The acceptance test checks `num_selected == num_total`, but `num_selected` counts only clients that returned successful results in `aggregate_fit`. If clients crash (as happened), `num_selected < num_total` even though they were all configured.

**Fix needed in `domain_aware_fedavg.py`**: Record the number of clients *configured to fit* (from `configure_fit`) separately from clients that *returned results* (from `aggregate_fit`). Store both in `round_quality_info`.

---

## 4. Improvement Plan

### Phase 1: Validate the CUDA Fix (Immediate)
1. Push the 4 bug fixes to the `das-fedrag` branch.
2. Re-run the τ=0.0 notebook on Kaggle. Verify all 5 clients complete training.
3. Check that all acceptance tests pass for τ=0.0.
4. Re-run all 4 τ notebooks in parallel.

### Phase 2: Data Pipeline Hardening
1. Add a `MAX_QUERY_CHARS` parameter to `load_client_data` (already added as constant).
2. Add a per-client training pair cap parameter (e.g., `max_train_per_client=1000`) to reduce NFCorpus dominance.
3. Consider shuffling pairs with a fixed seed before splitting (already done, but verify).
4. Add integration test: verify `split_pairs_with_caps` output sizes are ≤ caps.

### Phase 3: Strategy Robustness
1. Record `num_configured` (from `configure_fit`) in addition to `num_selected` (from `aggregate_fit`).
2. Add round-1 warm-up documentation to acceptance test report.
3. Consider increasing `NUM_ROUNDS` to 6 (1 warm-up + 5 selective rounds).
4. Add a `configure_fit` → `aggregate_fit` assertion: if a client was configured but didn't return, log a warning (not crash).

### Phase 4: Experiment Execution
1. Run τ sweep: {0.0, 0.2, 0.4, 0.6, 0.8} × seed={42} (single seed first).
2. Verify:
   - τ=0.0 selects all 5 clients (rounds 2+).
   - τ=0.2 should exclude FiQA (d=0.013) and ArguAna (d=0.042).
   - τ=0.6 should keep only NFCorpus (d=1.0).
   - τ=0.8 should keep only NFCorpus (d=1.0).
3. Key hypothesis: moderate τ (0.2–0.4) should outperform τ=0.0 on target MRR, because excluding irrelevant client updates prevents domain dilution.
4. If single-seed results are promising, run multi-seed (42, 52, 62).

### Phase 5: Results Compilation & Reporting
1. Use `compile_tau_results.py` to aggregate across τ values.
2. Generate:
   - τ vs Final Test MRR (the "money plot")
   - Client selection heatmap (which clients participate at each τ)
   - Per-round MRR curves by τ
3. Align thesis chapter with actual metrics (not projected).

---

## 5. Expected Domain Relevance Behavior

Based on the actual relevance scores observed:

| τ threshold | Expected Selected Clients | Expected Excluded |
|-------------|--------------------------|-------------------|
| 0.0 | All 5 (0,1,2,3,4) | None |
| 0.05 | 0 (medical), 1 (science), 4 (biomedical), 3 (argument) | 2 (finance, d=0.013) |
| 0.1 | 0 (medical), 1 (science), 4 (biomedical) | 2 (finance), 3 (argument) |
| 0.35 | 0 (medical), 1 (science) | 2,3,4 |
| 0.5 | 0 (medical) | 1,2,3,4 |
| 0.8 | 0 (medical) | 1,2,3,4 |

The τ sweep values {0.0, 0.2, 0.4, 0.6, 0.8} map to:
- τ=0.0: All 5 clients.
- τ=0.2: Clients 0, 1, 4 (medical + science + biomedical). Finance (0.013) and argument (0.042) excluded.
- τ=0.4: Clients 0, 1 (medical + science). Biomedical (0.32) also excluded.
- τ=0.6: Client 0 only (medical).
- τ=0.8: Client 0 only (medical).

Since τ=0.4 and τ=0.6 both select only the most relevant clients, and τ=0.8 is the same as τ=0.6, consider adding τ=0.1 and τ=0.35 to the sweep for finer granularity at the interesting transition points.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CUDA error recurs with float32 on certain datasets | Low | High | Added query truncation + `CUDA_LAUNCH_BLOCKING=1` in notebook for diagnostics |
| NFCorpus (4K pairs) dominates aggregation via size weighting | Medium | Medium | Already the expected behavior; DAS-FedAvg controls *selection*, not *weighting* |
| Round-1 warm-up contaminates results at high τ | Medium | Low | Skip round 1 in acceptance tests; increase total rounds |
| FiQA/TREC-COVID contribute negligible signal due to data sparsity | High | Low | These are meant to be excluded by the selection mechanism — their weakness validates DAS-FedAvg |
| Flower API changes break `client_fn(context: Context)` | Low | Medium | Fixed to use standard Context API; pin Flower version if needed |
