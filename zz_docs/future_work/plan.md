Here is the text converted into Markdown format, maintaining the original structure and content:

---

Grounding this in the thesis sources `report.tex`, `chapter3.tex`, `chapter4.tex`, and `chapter5.tex`, the intended design is:

* **QA-FedAvg:** blend dataset-size weights with inverse-loss quality scores, and recover plain FedAvg at $\alpha=0$.
* **DAS-FedAvg:** first select clients by domain-centroid similarity to a target domain, then apply QA-FedAvg on only the selected clients.

I also found a few concrete repo issues that should shape the plan:

* `run_alpha_sweep.py` points to a missing script (`federated_iid_qa.py`).
* `federated_noisy_qa.py` writes `noisy_results_alpha_*.csv`, but the checked-in files are `results_alpha_*.csv`, so the current saved CSVs are not aligned with the runnable pipeline.
* The current non-IID baseline in `federated_noniid_lsr.py` uses an NFCorpus knowledge store even for the SciFact client, which is important for both debugging QA-FedAvg and implementing domain-aware FL correctly.

### Objective 1 Plan: Fix the QA-FedAvg alpha-trend issue

**Establish a single source of truth for experiment outputs.**

* Standardize filenames, output directories, and CSV schema across runner and experiment scripts.
* Remove the mismatch between `results_alpha_*` and `noisy_results_alpha_*`.
* Add an experiment manifest per run containing alpha, seed, dataset split hash, noise ratio, round count, and code version so projected/manual CSVs cannot be mistaken for real runs.

**Audit the alpha-sweep execution path end to end.**

* Repair the sweep runner so it invokes the actual QA experiment script and reads the files that script truly produces.
* Ensure each alpha run starts from the same initial weights, same data split, same knowledge-store build, and same random seed.
* Verify that $\alpha=0.0$ reproduces FedAvg exactly, as required by the report.

**Add round-level instrumentation to prove whether alpha is influencing aggregation.**

* Log client ID, client dataset size, reported local loss, inverse-loss score, size weight, combined QA weight, and parameter-delta norm per round.
* Persist the exact server-side aggregation weights used in each round.
* Track model hashes before and after aggregation so we can confirm different alphas are actually producing different global models.

**Fix result attribution and ordering.**

* Stop assuming Flower returns clients in stable index order.
* Carry explicit client IDs through logging and CSV writing so “client 2 is noisy” remains true in recorded outputs.
* Make the noisy client identity explicit in the saved metadata instead of inferred from position.

**Revalidate the quality signal itself.**

* Confirm the loss used by `quality_aware_fedavg.py` is the real post-local-training LSR loss returned by the federated client, not a placeholder or mismatched metric.
* Check whether the current noise construction in `federated_noisy_qa.py` is too weak or misaligned with LSR, causing all clients to have similar effective training loss.
* If needed, redesign the noisy client generation so corruption is guaranteed to hurt the LSR objective, not just the surface (query, response) pairing.

**Strengthen the noisy-client benchmark so the expected trend becomes measurable.**

* Compare several corruption modes: within-client shuffled positives, cross-domain mismatched positives, random-document negatives, and mixed noise severity.
* Tune sensitivity if needed by increasing rounds, local epochs, or noise severity, because tiny local updates can wash out alpha differences even if the math is correct.
* Run multiple seeds and report mean and variance, not a single trajectory.

**Add post-aggregation evaluation, not just training-loss logging.**

* The report’s future scope already calls this out; it should become part of the QA-FedAvg validation now.
* After each round, evaluate the aggregated global retriever on held-out target-domain retrieval metrics: MRR, Recall@k, and NDCG@k.
* Accept the QA fix only if alpha trends appear in both aggregation weights and downstream retrieval performance, not only in projected loss tables.

**Define acceptance criteria for Objective 1.**

* $\alpha=0.0$ matches FedAvg.
* Higher-loss clients receive smaller weights when $\alpha>0$.
* Distinct alphas produce measurably distinct aggregation weights and model trajectories.
* The saved CSVs are regenerated from code and become reproducible artifacts, not projected placeholders.
* The final sweep either reproduces the intended inverted-U trend or gives evidence-backed reasons to revise the thesis claim.

---

### Objective 2 Plan: Implement Domain-Aware Federated RAG (DAS-FedAvg)

**Correct the data/knowledge-store architecture first.**

* Move from a single shared knowledge store toward per-client domain-aware stores, because DAS-FedAvg needs each client’s domain to be represented by its own document distribution.
* Keep a separate target-domain evaluation store for the domain you care about optimizing, initially medical/NFCorpus per the report.

**Implement domain representation generation.**

* For each client, compute a domain centroid as the mean embedding of that client’s knowledge-store documents, matching Chapter 5.
* Cache these centroids to disk so repeated experiments do not recompute them unnecessarily.
* Decide whether centroids are static from the base retriever or periodically refreshed; the initial implementation should use static cached centroids for stability and lower cost.

**Define the target-domain embedding clearly.**

* Use a centroid for the target domain built from the target dataset’s corpus or evaluation documents.
* For your current report-aligned setup, the first target domain should be NFCorpus/medical.
* Keep this configurable so later experiments can target SciFact, FiQA, or HotpotQA.

**Implement server-side domain-aware client selection.**

* In the federated strategy, compute cosine similarity between each client centroid and the target-domain centroid.
* Select only clients with similarity above threshold $\tau$, exactly as described in the thesis.
* Add a fallback policy so training does not stall if too few clients pass the threshold: top-k by similarity or minimum-selected-clients.

**Compose selection with quality-aware aggregation.**

* Build a new strategy class that performs two stages each round: domain-based participation filtering, then QA-FedAvg weighting over the selected subset.
* Keep $\alpha$ and $\tau$ independent so you can study domain selection and quality weighting separately.
* Preserve the FedAvg-compatible path for ablations: FedAvg, QA-FedAvg only, DAS only, and DAS+QA.

**Create the proper multi-domain benchmark.**

* Replace the current partially inconsistent non-IID setup with a true multi-domain federation where each client trains on its own domain data and retrieves against its own client store during local training.
* Start with the report’s medical-vs-science setup, then expand to more BEIR datasets once the core loop is stable.
* Save per-round selected-clients, similarity scores, and post-selection QA weights.

**Add evaluation specifically for DAS-FedAvg.**

* Evaluate global retrieval quality on the target domain after every round.
* Compare four systems: FedAvg, QA-FedAvg, DAS-FedAvg, and DAS+QA-FedAvg.
* Measure not just loss reduction but target-domain retrieval gains and whether irrelevant-domain clients are consistently excluded.

**Define acceptance criteria for Objective 2.**

* Domain centroids are reproducible and cached.
* Irrelevant clients fall below $\tau$ or are ranked lower by similarity.
* Selected-client sets are stable and interpretable.
* DAS improves target-domain retrieval compared with non-IID FedAvg, especially when irrelevant domains are present.
* DAS+QA outperforms either mechanism alone in the noisy plus cross-domain setting.

---

### Recommended implementation order

1. Fix experiment reproducibility and alpha-sweep plumbing.
2. Instrument and validate QA-FedAvg with real, regenerated outputs.
3. Repair the non-IID/domain setup so each client truly reflects its own domain.
4. Implement DAS client centroids and target-domain centroid.
5. Implement server-side selection with $\tau$.
6. Compose DAS with QA-FedAvg and run ablations.
7. Add post-training retrieval evaluation and thesis-ready result export.

---

Would you like me to create that phase-by-phase execution checklist mapped to the specific files mentioned?