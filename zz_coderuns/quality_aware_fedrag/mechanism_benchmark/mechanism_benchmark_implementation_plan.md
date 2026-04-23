**Mechanism Track Goal**

Build a clean, controlled benchmark in `q-fedrag2` that proves:

- standard `FedAvg` is vulnerable to client-quality heterogeneity
- `QA-FedRAG` fixes that vulnerability
- the improvement is visible on final retrieval metrics, not just on internal weights

This track is not trying to maximize realism first.
It is trying to prove the proposition clearly.

---

**Target Benchmark Design**

Use a single-domain federation on `NFCorpus` with:

- `5` clients
- equal client sizes
- one cleanly controlled quality ladder across clients
- fixed generator
- retrieval-first evaluation
- alphas: `0.0`, `0.3`, `0.7`, `1.0`

Recommended client-quality structure:

- client `0`: clean
- client `1`: clean
- client `2`: clean
- client `3`: mild noise
- client `4`: strong noise

Recommended noise mode:

- primary mechanism benchmark: `shuffle`

Recommended noise ratios:

- client `3`: `0.3`
- client `4`: `0.7`

Rationale:

- equal client sizes isolate quality heterogeneity
- multiple noise levels are stronger evidence than one noisy client
- `shuffle` gives the cleanest signal that QA-FedRAG should exploit

---

**Implementation Plan**

**Phase 1: Freeze the mechanism-track scope**
Files:
- [federated_noisy_qa.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_noisy_qa.py)
- [run_alpha_sweep.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/run_alpha_sweep.py)
- [single_alpha_kaggle_colab.ipynb](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/single_alpha_kaggle_colab.ipynb)

Changes:

- define an explicit `mechanism` experiment mode
- make that mode the main path in `q-fedrag2`
- keep the current `stress` logic available, but secondary

What this means operationally:

- the mechanism mode should always use:
  - equal split
  - 5 clients
  - shuffle corruption
  - fixed per-client noise schedule
  - fixed seed during first-pass runs

Rationale:

- this prevents the branch from drifting into “many mixed experiments”
- makes the benchmark reproducible and easy to explain

---

**Phase 2: Replace “one noisy client” with a quality ladder**
File:
- [federated_noisy_qa.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_noisy_qa.py)

Changes:

- stop treating client `4` as the only noisy client in mechanism mode
- introduce a per-client corruption schedule, for example:
  - `0: 0.0`
  - `1: 0.0`
  - `2: 0.0`
  - `3: 0.3`
  - `4: 0.7`

Implementation idea:

- replace the current `noisy_client_id + noise_ratio` logic in mechanism mode with a `client_noise_map`
- keep the old single-noisy-client path for stress mode

Rationale:

- this gives QA-FedRAG a more realistic aggregation problem than “one obviously bad client”
- it also makes alpha effects smoother and easier to observe
- FedAvg should be systematically pulled away by lower-quality clients, while QA-FedRAG should correct that

---

**Phase 3: Keep corruption mode simple and diagnosis-friendly**
File:
- [federated_noisy_qa.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_noisy_qa.py)

Changes:

- use `shuffle` only for mechanism mode
- do not use `hard_negative` in this track
- keep `hard_negative` only for robustness mode later

Rationale:

- the mechanism track must answer one question:
  - “does quality-aware aggregation help when low-quality clients are clearly low-quality?”
- `shuffle` is the cleanest corruption mode for that
- if you mix harder semantic corruption here, you reintroduce ambiguity too early

---

**Phase 4: Keep client sizes exactly equal**
File:
- [federated_noisy_qa.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_noisy_qa.py)

Changes:

- enforce `CLIENT_SPLIT_MODE = "equal"` in mechanism mode
- remove `NOISY_CLIENT_DATA_FRACTION` from the mechanism benchmark path

Rationale:

- size imbalance introduces a second confound
- if the purpose is to prove QA-FedRAG, the cleanest story is:
  - all clients are same size
  - only quality differs
  - FedAvg fails because it cannot distinguish client quality
  - QA-FedRAG improves because it can

---

**Phase 5: Use a weaker retriever for this branch**
File:
- [prepare_beir_data.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/prepare_beir_data.py)

Changes:

- introduce a configurable retriever model name
- set mechanism mode to use a weaker retriever than the current default
- keep the stronger retriever available for later robustness tests

Branch-level rule:

- `q-fedrag2` mechanism benchmark should default to the weaker retriever

Rationale:

- current retrieval headroom is too small
- the current pretrained retriever is already strong enough that alpha changes do not easily move rankings
- using a weaker retriever increases measurable learning headroom

Important framing:

- this is not “cheating”
- this is a proof-of-mechanism benchmark choice
- later, robustness can be checked with the stronger retriever

---

**Phase 6: Preserve the current good parts of the pipeline**
Files:
- [quality_aware_fedavg.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/quality_aware_fedavg.py)
- [federated_noisy_qa.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_noisy_qa.py)

Keep:

- shared rank-based quality signal
- softmax-over-zscore weighting with `quality_beta`
- server-validation best-round selection
- final-test reporting
- detailed client-level instrumentation

Rationale:

- these are already the strongest parts of the current code
- the redesign should change the benchmark, not throw away the improvements that made the pipeline mechanically sound

---

**Phase 7: Simplify the metrics you report**
Files:
- [compile_alpha_results.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/compile_alpha_results.py)
- [run_alpha_sweep.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/run_alpha_sweep.py)

Main reported metrics:

- final test `MRR`
- final test `NDCG@k`

Secondary metrics:

- noisy / lower-quality client weights by round
- shared rank-displacement quality loss by round
- best validation round

De-emphasize:

- `avg_train_loss`
- `avg_loss` as the headline result

Rationale:

- the paper claim is about improving retrieval quality, not minimizing local loss
- your current history already shows that training loss can move without meaningful retrieval gain

---

**Phase 8: Define the exact first-pass hyperparameter regime**
Files:
- [federated_noisy_qa.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_noisy_qa.py)
- [single_alpha_kaggle_colab.ipynb](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/single_alpha_kaggle_colab.ipynb)

Recommended first-pass settings:

- clients: `5`
- alphas: `0.0`, `0.3`, `0.7`, `1.0`
- rounds: `4`
- local epochs: `1`
- learning rate: keep conservative initially
- split: equal
- benchmark mode: mechanism
- corruption: shuffle
- noise schedule:
  - `0, 0, 0, 0.3, 0.7`
- seed: `42`

Rationale:

- keep optimization pressure low because earlier runs showed that more aggressive training can collapse the retriever or flatten the effect
- the redesign should test benchmark clarity before testing optimization scale

---

**Phase 9: Add explicit mechanism-track acceptance criteria**
Files:
- [federated_noisy_qa.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/federated_noisy_qa.py)
- optionally [run_alpha_sweep.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/run_alpha_sweep.py)

Mechanism-track run is acceptable only if:

- lower-quality clients receive systematically lower weights than clean clients
- `alpha=0.0` behaves like pure FedAvg
- at least one nonzero alpha beats `alpha=0.0` on final test MRR or NDCG
- `alpha=1.0` is not always best, so there is a real tradeoff
- best-round selection does not collapse all alphas to an identical outcome

Rationale:

- these acceptance checks are closer to the real proposition than the current generic checks
- they tell you whether the benchmark is actually usable for a paper

---

**Phase 10: Add a second mechanism benchmark only after the first one works**
After the first mechanism benchmark works, add one follow-up:

Mechanism benchmark B:

- same setup
- stronger noise ladder, for example:
  - `0, 0, 0.2, 0.5, 0.8`

Rationale:

- this checks whether the win survives stronger quality heterogeneity
- but only do this after benchmark A gives a clean result
- otherwise you risk debugging two benchmarks at once

---

**Phase 11: Multi-seed only after one-seed success**
Files:
- [run_alpha_sweep.py](/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/quality_aware_fedrag/run_alpha_sweep.py)

First:

- use `seed=42` to validate the benchmark design

Only after visible alpha separation:

- run `42`, `52`, `62`

Rationale:

- one-seed runs are for discovering whether the benchmark is promising
- multi-seed runs are for validating a benchmark, not inventing it

---

**Recommended Implementation Order**

1. Add explicit `mechanism` mode in the QA experiment driver.
2. Add per-client corruption schedule support.
3. Lock mechanism mode to `equal` split and `shuffle` corruption.
4. Make retriever model configurable and switch mechanism mode to a weaker retriever.
5. Update notebook defaults to the mechanism benchmark.
6. Update result compilation so final-test MRR/NDCG are the main outputs.
7. Run one-seed alpha sweep.
8. Review whether the alpha curve is finally visible.
9. If yes, run multiple seeds.
10. Only after that, define the robustness track in `q-fedrag2`.

---

**What I expect if this plan is right**

In the first successful mechanism benchmark, you should see:

- `alpha=0.0` underperforming because it averages clean and noisy clients blindly
- `alpha=0.3` or `alpha=0.7` doing best
- `alpha=1.0` either slightly worse than the best midpoint or unstable/overselective
- client `4` consistently receiving the lowest weight
- client `3` also somewhat downweighted relative to clean clients
- visible separation in final test MRR/NDCG, not just internal weights

That would be the first real proof that `QA-FedRAG` is not only logically correct, but effectively better than the baseline.
