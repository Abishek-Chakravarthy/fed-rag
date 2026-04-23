
Goal

Make QA-FedAvg produce a defensible, visible improvement over FedAvg by fixing three things in order:

the quality signal
round selection / evaluation design
benchmark scale and robustness
The current code is mechanically working, but the signal is too weak and too smooth, so alpha changes weights without changing retrieval enough.

Phase 1: Fix The Quality Signal

1. Replace the current shared quality loss with a rank-based signal
Target file:
quality_aware_fedavg.py
federated_noisy_qa.py

What to change:

Stop using -log(0.5 * (shared_mrr + shared_ndcg)) as the main aggregation loss.
For each shared probe pair, compute:
rank of the correct document before local training
rank of the correct document after local training
Define a per-client quality signal from:
average rank delta, or
average correct-document displacement, or
fraction of probes where the true doc rank improved vs worsened
Recommended primary signal:

quality_loss = mean(max(rank_after - rank_before, 0))
lower is better
clients that push the true document downward get penalized directly
Why:

this is much more sensitive to hard-negative corruption than aggregate MRR/NDCG
it measures exactly the failure mode you care about
Implementation steps:

add a helper in federated_noisy_qa.py to compute correct-doc ranks on shared probes
cache the “before local training” ranks at the start of each client round
compute “after local training” ranks after local training
store both raw rank stats and aggregated quality loss in client metrics
keep shared MRR/NDCG as diagnostics, not the server weighting input
2. Keep per-client holdouts, but diagnostic only
Target file:
federated_noisy_qa.py

What to change:

preserve current per-client holdout metrics
do not use them for weighting
write them into CSV/manifest for interpretability
Why:

useful for analysis
but still not fair for cross-client weighting
3. Sharpen the weighting transform on the server
Target file:
quality_aware_fedavg.py

What to change:

replace plain inverse-loss quality score with a sharper normalized mapping
recommended formula:
compute z-score of client losses within the round
quality_scores = softmax(-beta * zscores)
keep alpha interpolation with size weights
add beta as a strategy parameter, default around 5.0
Why:

preserves your QA-FedAvg design
avoids replacing the algorithm with a hard exclusion heuristic
makes small but consistent loss differences matter more
Acceptance condition for this phase:

on the shared comparison signal, the noisy client is among the worst clients in most rounds
alpha visibly changes client weights more than it does now
Phase 2: Fix Evaluation Design

4. Split data into three evaluation roles
Target file:
prepare_beir_data.py
federated_noisy_qa.py

What to change:

Instead of one eval stream plus shared-quality pairs, produce:
shared_quality_pairs
server_val_pairs
final_test_pairs
Recommended sizes after scaling:

shared_quality_pairs = 400
server_val_pairs = 400
final_test_pairs = 500
Why:

shared_quality_pairs should be used only for client weighting
server_val_pairs should be used only for best-round selection
final_test_pairs should be untouched until final reporting
Implementation steps:

update setup_dataset() to return the 3 pair groups
update manifest writing so all split counts and hashes are recorded
update CSV schema to include both validation and final test metrics
5. Add best-round selection
Target file:
federated_noisy_qa.py

What to change:

after each aggregated round, evaluate on server_val_pairs
track the best round using:
primary: server_val_mrr
secondary tie-breaker: server_val_ndcg_at_k
save the best aggregated model weights in memory
after training ends, evaluate that best model once on final_test_pairs
Why:

current runs often peak before the last round
this prevents late-round degradation from weakening the reported result
Outputs to add:

best_round
best_val_mrr
best_val_ndcg
final_test_mrr
final_test_ndcg
Acceptance condition for this phase:

results no longer depend on “last round happened to be good”
reported final metric is validation-selected, not arbitrary
Phase 3: Strengthen The Benchmark

6. Add a control benchmark with shuffle noise
Target file:
federated_noisy_qa.py
run_alpha_sweep.py

What to change:

define two official experiment modes:
control: shuffle
stress: hard_negative
expose them cleanly in the sweep summary
Why:

shuffle should verify the mechanism on easier noise
hard_negative should test whether it still works under subtle semantic corruption
Interpretation:

if QA-FedAvg works on shuffle but not hard_negative, the method is plausible but the metric is still too weak for subtle corruption
if it works on both, you have a much stronger contribution
7. Strengthen hard_negative construction
Target file:
federated_noisy_qa.py

What to change:

instead of sampling randomly from top-5 wrong docs, choose from the top-ranked wrong docs more consistently
recommended:
retrieve top N
exclude the true doc
choose the highest-ranked wrong doc, or sample from top-2 only
keep noise in-domain
Why:

current hard-negative corruption may still be too semantically acceptable
you want the wrong signal to be consistently harmful, not randomly weak
8. Increase client count
Target file:
federated_noisy_qa.py

What to change:

move from 3 to 5 clients
Recommended setups:

control benchmark:
5 equal clients
stress benchmark:
0.15 / 0.15 / 0.15 / 0.15 / 0.40 with one noisy client
Why:

5 clients gives smoother aggregation behavior
one noisy client among more clean clients is more realistic and more informative than a tiny 3-client federation
Acceptance condition for this phase:

noisy client remains identifiable
alpha changes have measurable downstream effect in at least the control setting
Phase 4: Scale The Data Carefully

9. Increase training and corpus size without increasing optimization pressure too much
Target file:
prepare_beir_data.py
federated_noisy_qa.py

Recommended new defaults:

MAX_TRAIN = 4000 initially
MAX_DOCS = 8000
shared_quality_pairs = 400
server_val_pairs = 400
final_test_pairs = 500
Keep optimization conservative:

LEARNING_RATE = 2e-6
LOCAL_EPOCHS = 1
NUM_ROUNDS = 4
Why:

more data makes retrieval metrics less noisy
more docs makes retrieval more realistic
but you must not repeat the aggressive setup that caused collapse in earlier runs
Only after stable results:

test NUM_ROUNDS = 6
do not increase epochs first
10. Consider a second dataset only after NFCorpus is working
Target file:
prepare_beir_data.py

What to change:

keep NFCorpus as the main benchmark first
once the method works there, repeat on one more BEIR dataset with enough positives
Why:

changing dataset now may confuse debugging with generalization
better to first get one clean success story, then show transfer
Phase 5: Make The Results Thesis-Ready

11. Update result schemas and summaries
Target file:
federated_noisy_qa.py
run_alpha_sweep.py
compile_alpha_results.py

Add to CSV/summary:

shared_quality_loss
shared_rank_delta
server_val_mrr
server_val_ndcg
best_round
final_test_mrr
final_test_ndcg
noisy-client weight by round
Plots to generate:

alpha vs final test MRR
alpha vs final test NDCG
round vs validation MRR for each alpha
noisy client weight vs round
rank-delta quality score vs round by client
12. Add multi-seed support as the final validation stage
Target file:
run_alpha_sweep.py

What to change:

run seeds like 42, 52, 62
optionally rotate noisy client ID per seed group
aggregate mean and std on:
best validation MRR
final test MRR
final test NDCG
Why:

one seed is enough for fast screening
not enough for final claims
Recommended Execution Order

Implement shared rank-delta quality metric.
Implement sharper server weighting with temperature beta.
Add server_val and final_test splits.
Add best-round selection.
Run shuffle control with 5 clients and scaled data.
If control works, run hard_negative stress test.
Run alpha sweep for 0.0, 0.3, 0.7, 1.0.
Run 3 seeds.
Only then consider a second dataset.
Concrete First Target Configuration

For the first serious rerun after implementation:

clients: 5
alphas: 0.0, 0.3, 0.7, 1.0
learning rate: 2e-6
local epochs: 1
rounds: 4
max train: 4000
max docs: 8000
shared quality: 400
server val: 400
final test: 500
control: shuffle, noise_ratio=0.7, equal split
stress: hard_negative, noise_ratio=0.8, noisy client fraction 0.4
Success Criteria

You should treat the new setup as successful only if:

the noisy client is consistently among the worst on the rank-based shared signal
alpha visibly changes client weights
at least one nonzero alpha beats alpha=0.0 on final test MRR/NDCG in the control benchmark
alpha=1.0 is not always best, so there is a real tradeoff
the trend remains under multiple seeds
If you want, I can next turn this into a file-by-file checklist with exact functions/classes to add or modify in each Python file.