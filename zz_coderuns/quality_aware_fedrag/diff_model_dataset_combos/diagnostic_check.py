"""
Non-Federated Diagnostic: Does contrastive training on FiQA improve retrieval?

Trains a single retriever on a clean dataset (no federation, no noise, no Flower)
using MultipleNegativesRankingLoss (InfoNCE) and measures MRR / NDCG before and
after.  Sweeps multiple learning-rate and dataset variants to find the winning
config before investing in full federated runs.

Gates for Phase 1 (QA-FedAvg) and Phase 1B (DAS-FedAvg):
  PASS  → any variant shows ≥ +5% relative test MRR improvement
  MARGINAL → +1 to +5 % — consider D3 (more data) or CQADupStack
  FAIL  → all variants ≤ 0 % — FiQA is at ceiling, switch dataset

Variants:
  D1  fiqa     LR=2e-6  max_train=4000   (matches proven QA-FedAvg config)
  D2  fiqa     LR=5e-7  max_train=4000   (matches proven DAS-FedAvg config)
  D3  fiqa     LR=2e-6  max_train=10000  (use more of FiQA's 14K pairs)
  D4  fiqa     LR=1e-6  max_train=4000   (untested midpoint)
  D5  nfcorpus LR=2e-6  max_train=4000   (control — must reproduce +5.6%)

Usage (Kaggle / lfed-rag/zz_coderuns/quality_aware_fedrag/diff_model_dataset_combos/diagnostic_kaggle.ipynbocal):
    python fiqa_diagnostic.py                   # run all variants sequentially
    python fiqa_diagnostic.py --variants D1,D5  # run specific variants
    python fiqa_diagnostic.py --run-single D1   # internal: one variant in-process
"""

import argparse
import json
import os
import subprocess
import sys

import torch

torch.set_num_threads(1)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
MAX_DOCS = 8000
MAX_SERVER_VAL = 400
MAX_FINAL_TEST = 500
MAX_SHARED_QUALITY = 0   # Not needed for diagnostic

PASS_THRESHOLD_PCT = 5.0     # Relative MRR improvement % to count as PASS
MARGINAL_THRESHOLD_PCT = 1.0 # Below this → FAIL

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diagnostic_results"
)

ALL_VARIANTS = {
    #  ID      dataset      LR       max_train  notes
    "D1": {"dataset": "fiqa",     "lr": 2e-6, "max_train": 4000,
           "note": "FiQA @ LR=2e-6 (proven QA-FedAvg config)"},
    "D2": {"dataset": "fiqa",     "lr": 5e-7, "max_train": 4000,
           "note": "FiQA @ LR=5e-7 (proven DAS-FedAvg config)"},
    "D3": {"dataset": "fiqa",     "lr": 2e-6, "max_train": 10000,
           "note": "FiQA @ LR=2e-6, 10K pairs (more signal)"},
    "D4": {"dataset": "fiqa",     "lr": 1e-6, "max_train": 4000,
           "note": "FiQA @ LR=1e-6 (untested midpoint)"},
    "D5": {"dataset": "nfcorpus", "lr": 2e-6, "max_train": 4000,
           "note": "NFCorpus control — must reproduce +5.6%"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def verdict(delta_pct: float) -> str:
    if delta_pct >= PASS_THRESHOLD_PCT:
        return "✅ PASS"
    if delta_pct >= MARGINAL_THRESHOLD_PCT:
        return "⚠️  MARGINAL"
    return "❌ FAIL"


# ---------------------------------------------------------------------------
# Single-variant runner (called in-process via --run-single)
# ---------------------------------------------------------------------------

def run_single_variant(vid: str, dataset: str, lr: float, max_train: int):
    """
    Train one variant end-to-end in the current process.

    We import heavy dependencies here (inside the function) so that the
    subprocess-dispatch path doesn't pay the import cost.
    """
    from datasets import Dataset
    from datasets.utils import logging as datasets_logging
    from sentence_transformers import SentenceTransformerTrainingArguments
    from transformers.utils import logging as transformers_logging

    from contrastive_trainer import ContrastiveFlowerClient, ContrastiveRetrieverTrainer, _get_weights
    from prepare_beir_data import (
        setup_dataset,
        evaluate_retriever,
        TOP_K,
        SEED,
    )

    datasets_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()

    cfg = ALL_VARIANTS[vid]
    print(f"\n{'=' * 72}")
    print(f"Variant {vid}: {cfg['note']}")
    print(f"  dataset={dataset}  lr={lr:.0e}  max_train={max_train}  max_docs={MAX_DOCS}")
    print(f"  device={get_runtime_device()}")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data = setup_dataset(
        dataset,
        max_train=max_train,
        max_server_val=MAX_SERVER_VAL,
        max_final_test=MAX_FINAL_TEST,
        max_shared_quality=MAX_SHARED_QUALITY,
        max_docs=MAX_DOCS,
        seed=SEED,
    )

    retriever      = data["retriever"]
    knowledge_store = data["knowledge_store"]
    train_pairs    = data["train_pairs"]
    server_val_pairs = data["server_val_pairs"]
    final_test_pairs = data["final_test_pairs"]

    print(f"\n  Dataset stats:")
    print(f"    train pairs    : {len(train_pairs)}")
    print(f"    server val     : {len(server_val_pairs)}")
    print(f"    final test     : {len(final_test_pairs)}")
    print(f"    knowledge store: {knowledge_store.count} docs")

    if len(train_pairs) == 0:
        print("  ❌ ERROR: No training pairs built — check dataset name and qrels.")
        return None
    if len(final_test_pairs) == 0:
        print("  ❌ ERROR: No test pairs built — knowledge store may be too small.")
        return None

    # ------------------------------------------------------------------
    # 2. Pre-train evaluation
    # ------------------------------------------------------------------
    pre_val  = evaluate_retriever(retriever, knowledge_store, server_val_pairs,  top_k=TOP_K)
    pre_test = evaluate_retriever(retriever, knowledge_store, final_test_pairs,  top_k=TOP_K)

    print(f"\n  PRE-TRAIN  val  MRR={pre_val['mrr']:.6f}  "
          f"NDCG={pre_val['ndcg_at_k']:.6f}  Recall={pre_val['recall_at_k']:.4f}")
    print(f"  PRE-TRAIN  test MRR={pre_test['mrr']:.6f}  "
          f"NDCG={pre_test['ndcg_at_k']:.6f}  Recall={pre_test['recall_at_k']:.4f}")

    # ------------------------------------------------------------------
    # 3. Build training dataset (same column schema as federated scripts)
    # ------------------------------------------------------------------
    train_dataset = Dataset.from_dict({
        "query":    [p["query"]    for p in train_pairs],
        "response": [p["response"] for p in train_pairs],
    })

    # ------------------------------------------------------------------
    # 4. Set up trainer (direct training — no Flower, no simulation)
    #    We reuse ContrastiveFlowerClient internals but call trainer.train()
    #    directly instead of going through .fit() / .to_client().
    # ------------------------------------------------------------------
    tmp_dir = os.path.join(OUTPUT_DIR, f"tmp_{vid}")
    os.makedirs(tmp_dir, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    # Pull the query encoder from the retriever (same object the retriever uses
    # for embedding, so evaluate_retriever() sees the updated weights).
    model = retriever.query_encoder if retriever.query_encoder else retriever.encoder

    # Build a throwaway ContrastiveFlowerClient just to get the renamed dataset
    # and the loss — then train directly without Flower.
    client = ContrastiveFlowerClient(
        model=model,
        train_dataset=train_dataset,
        training_args=training_args,
    )

    trainer = ContrastiveRetrieverTrainer(
        model=model,
        args=training_args,
        train_dataset=client.train_dataset,  # already renamed anchor/positive
        loss=client.loss,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"\n  Training on {len(train_dataset)} pairs, lr={lr:.0e}, 1 epoch ...")
    train_output = trainer.train()
    print(f"  Training loss: {train_output.training_loss:.6f}")

    # ------------------------------------------------------------------
    # 6. Post-train evaluation
    # ------------------------------------------------------------------
    post_val  = evaluate_retriever(retriever, knowledge_store, server_val_pairs,  top_k=TOP_K)
    post_test = evaluate_retriever(retriever, knowledge_store, final_test_pairs,  top_k=TOP_K)

    print(f"\n  POST-TRAIN val  MRR={post_val['mrr']:.6f}  "
          f"NDCG={post_val['ndcg_at_k']:.6f}  Recall={post_val['recall_at_k']:.4f}")
    print(f"  POST-TRAIN test MRR={post_test['mrr']:.6f}  "
          f"NDCG={post_test['ndcg_at_k']:.6f}  Recall={post_test['recall_at_k']:.4f}")

    # ------------------------------------------------------------------
    # 7. Compute deltas and verdict
    # ------------------------------------------------------------------
    val_delta_pct  = (post_val["mrr"]  - pre_val["mrr"])  / max(pre_val["mrr"],  1e-9) * 100
    test_delta_pct = (post_test["mrr"] - pre_test["mrr"]) / max(pre_test["mrr"], 1e-9) * 100
    val_ndcg_delta_pct  = (post_val["ndcg_at_k"]  - pre_val["ndcg_at_k"])  / max(pre_val["ndcg_at_k"],  1e-9) * 100
    test_ndcg_delta_pct = (post_test["ndcg_at_k"] - pre_test["ndcg_at_k"]) / max(pre_test["ndcg_at_k"], 1e-9) * 100

    v = verdict(test_delta_pct)
    print(f"\n  delta val_MRR={val_delta_pct:+.2f}%  delta test_MRR={test_delta_pct:+.2f}%  → {v}")
    print(f"  delta val_NDCG={val_ndcg_delta_pct:+.2f}%  delta test_NDCG={test_ndcg_delta_pct:+.2f}%")

    # ------------------------------------------------------------------
    # 8. Save result JSON
    # ------------------------------------------------------------------
    result = {
        "id":                  vid,
        "note":                cfg["note"],
        "dataset":             dataset,
        "lr":                  lr,
        "max_train":           max_train,
        "max_docs":            MAX_DOCS,
        "train_pairs_built":   len(train_pairs),
        "server_val_pairs":    len(server_val_pairs),
        "final_test_pairs":    len(final_test_pairs),
        "training_loss":       float(train_output.training_loss),
        "pre_val_mrr":         pre_val["mrr"],
        "post_val_mrr":        post_val["mrr"],
        "val_mrr_delta_pct":   val_delta_pct,
        "pre_test_mrr":        pre_test["mrr"],
        "post_test_mrr":       post_test["mrr"],
        "test_mrr_delta_pct":  test_delta_pct,
        "pre_val_ndcg":        pre_val["ndcg_at_k"],
        "post_val_ndcg":       post_val["ndcg_at_k"],
        "val_ndcg_delta_pct":  val_ndcg_delta_pct,
        "pre_test_ndcg":       pre_test["ndcg_at_k"],
        "post_test_ndcg":      post_test["ndcg_at_k"],
        "test_ndcg_delta_pct": test_ndcg_delta_pct,
        "pre_test_recall":     pre_test["recall_at_k"],
        "post_test_recall":    post_test["recall_at_k"],
        "verdict":             v,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(OUTPUT_DIR, f"result_{vid}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result saved → {result_path}")

    return result


# ---------------------------------------------------------------------------
# Subprocess dispatcher
# Each variant runs as its own subprocess to get a clean AcceleratorState and
# avoid GPU memory fragmentation between variants.
# ---------------------------------------------------------------------------

def run_variant_subprocess(vid: str) -> bool:
    script_path = os.path.abspath(__file__)
    cmd = [sys.executable, "-u", script_path, "--run-single", vid]

    print(f"\n{'─' * 72}")
    print(f"Launching variant {vid} as subprocess ...")
    print(f"{'─' * 72}")

    process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(script_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode == 0


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    col_w = 42
    print(f"\n\n{'=' * 100}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 100}")
    header = (
        f"{'ID':<4}  {'Dataset':<10}  {'LR':<7}  {'Train':<6}  "
        f"{'Pre MRR':>9}  {'Post MRR':>9}  {'ΔMRR%':>7}  "
        f"{'Pre NDCG':>9}  {'Post NDCG':>9}  {'ΔNDCG%':>7}  Verdict"
    )
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r['id']:<4}  {r['dataset']:<10}  {r['lr']:<7.0e}  {r['max_train']:<6}  "
            f"{r['pre_test_mrr']:>9.6f}  {r['post_test_mrr']:>9.6f}  {r['test_mrr_delta_pct']:>+6.1f}%  "
            f"{r['pre_test_ndcg']:>9.6f}  {r['post_test_ndcg']:>9.6f}  {r['test_ndcg_delta_pct']:>+6.1f}%  "
            f"{r['verdict']}"
        )

    # Gate evaluation
    passing   = [r for r in results if r["test_mrr_delta_pct"] >= PASS_THRESHOLD_PCT]
    marginals = [r for r in results if MARGINAL_THRESHOLD_PCT <= r["test_mrr_delta_pct"] < PASS_THRESHOLD_PCT]
    fails     = [r for r in results if r["test_mrr_delta_pct"] < MARGINAL_THRESHOLD_PCT]

    print(f"\n{'=' * 100}")
    print("GATE RESULT")
    print(f"{'=' * 100}")

    if passing:
        best = max(passing, key=lambda r: r["test_mrr_delta_pct"])
        print(f"✅  GATE PASSED — {len(passing)} variant(s) exceed {PASS_THRESHOLD_PCT:.0f}% MRR improvement.")
        print(f"    Best variant: {best['id']}  ({best['note']})")
        print(f"    LR={best['lr']:.0e}, max_train={best['max_train']}")
        print(f"    Use this config for both Phase 1A (QA-FedAvg) and Phase 1B (DAS-FedAvg).")
        print(f"\n    ► QA-FedAvg:  set DATASET_NAME='fiqa' and LEARNING_RATE={best['lr']:.0e} in federated_noisy_qa.py")
        print(f"    ► DAS-FedAvg: set TARGET_DATASET='fiqa' and LEARNING_RATE={best['lr']:.0e} in federated_das.py")
    elif marginals:
        best = max(marginals, key=lambda r: r["test_mrr_delta_pct"])
        print(f"⚠️   GATE MARGINAL — best improvement is {best['test_mrr_delta_pct']:+.1f}% (below {PASS_THRESHOLD_PCT:.0f}% threshold).")
        print(f"    Variant {best['id']} ({best['note']}) is the best candidate.")
        if "D3" not in [r["id"] for r in results]:
            print(f"    Suggestion: run variant D3 (FiQA, 10K pairs) — more data may push past threshold.")
        else:
            print(f"    Suggestion: try CQADupStack (android subforum) — more niche domain, more headroom.")
    else:
        print(f"❌  GATE FAILED — no variant improved MRR on FiQA.")
        print(f"    Suggestions:")
        print(f"      1. Check D5 (NFCorpus control) — if it also fails, there's an environment issue.")
        print(f"      2. Switch to CQADupStack: replace 'fiqa' with 'BeIR/cqadupstack/android' in this script.")
        print(f"      3. Investigate training loss: is it decreasing? If not, raise LR.")

    # Control check
    d5 = next((r for r in results if r["id"] == "D5"), None)
    if d5:
        print(f"\n  Control (D5/NFCorpus): {d5['test_mrr_delta_pct']:+.1f}% — ", end="")
        if d5["test_mrr_delta_pct"] > 4:
            print("✅ matches known +5.6% result. Environment is healthy.")
        elif d5["test_mrr_delta_pct"] > 0:
            print("⚠️  lower than expected +5.6% — minor variance is OK.")
        else:
            print("❌ degraded! Environment issue — check GPU, deps, branch.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Non-federated diagnostic: does contrastive training on FiQA improve retrieval?"
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(ALL_VARIANTS.keys()),
        help=f"Comma-separated variant IDs to run. Default: all ({', '.join(ALL_VARIANTS.keys())})",
    )
    parser.add_argument(
        "--run-single",
        type=str,
        default=None,
        metavar="ID",
        help="Internal flag: run exactly one variant in this process then exit.",
    )
    args = parser.parse_args()

    # ── Internal path: called by subprocess dispatcher ──────────────────────
    if args.run_single:
        vid = args.run_single
        if vid not in ALL_VARIANTS:
            print(f"ERROR: unknown variant '{vid}'. Available: {list(ALL_VARIANTS.keys())}")
            sys.exit(1)
        cfg = ALL_VARIANTS[vid]
        result = run_single_variant(vid, cfg["dataset"], cfg["lr"], cfg["max_train"])
        sys.exit(0 if result is not None else 1)

    # ── Orchestrator path: launch each selected variant as a subprocess ──────
    selected = [v.strip() for v in args.variants.split(",") if v.strip() in ALL_VARIANTS]
    unknown  = [v.strip() for v in args.variants.split(",") if v.strip() not in ALL_VARIANTS]
    if unknown:
        print(f"WARNING: ignoring unknown variant IDs: {unknown}")
    if not selected:
        print("ERROR: no valid variants selected.")
        sys.exit(1)

    print(f"Running variants: {selected}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for vid in selected:
        success = run_variant_subprocess(vid)
        if not success:
            print(f"  ⚠️  Variant {vid} subprocess returned non-zero exit code — check logs above.")

    # Collect JSON results written by each subprocess
    results = []
    for vid in selected:
        result_path = os.path.join(OUTPUT_DIR, f"result_{vid}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            print(f"  ⚠️  No result file found for variant {vid} — it may have crashed.")

    if results:
        print_summary(results)

        summary_path = os.path.join(OUTPUT_DIR, "diagnostic_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull summary written → {summary_path}")
    else:
        print("\n❌ No results to summarize. All variants failed.")


if __name__ == "__main__":
    main()
