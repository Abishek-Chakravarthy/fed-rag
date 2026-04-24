"""
Step 1 Diagnostic: Can LSR training improve retrieval at all?

Trains a single retriever on clean NFCorpus data (no federation, no noise)
and measures MRR/NDCG before and after. Sweeps retriever model and LR.

Usage:
    python diagnostic_single_client.py
    python diagnostic_single_client.py --variants 1a,1c      # run specific variants only
"""

import torch
torch.set_num_threads(1)

import argparse
import json
import os
import sys

from datasets import Dataset
from datasets.utils import logging as datasets_logging
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import GenerationConfig
from transformers.utils import logging as transformers_logging
from accelerate.state import AcceleratorState

from fed_rag import RAGSystem, RAGConfig
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.trainers import HuggingFaceTrainerForLSR

from prepare_beir_data import (
    setup_dataset,
    evaluate_retriever,
    TOP_K,
    SEED,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
datasets_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

GENERATOR_MODEL = "distilgpt2"
BATCH_SIZE = 8
MAX_DOCS = 8000
MAX_TRAIN = 2000
MAX_SERVER_VAL = 400
MAX_FINAL_TEST = 400
MAX_SHARED_QUALITY = 0

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diagnostic_results"
)

ALL_VARIANTS = {
    "1a": {"retriever": "sentence-transformers/paraphrase-MiniLM-L3-v2", "lr": 2e-6},
    "1b": {"retriever": "sentence-transformers/paraphrase-MiniLM-L3-v2", "lr": 5e-7},
    "1c": {"retriever": "sentence-transformers/all-MiniLM-L6-v2",        "lr": 2e-6},
    "1d": {"retriever": "sentence-transformers/all-MiniLM-L6-v2",        "lr": 5e-7},
}


def get_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_generator_load_kwargs() -> dict:
    device = get_runtime_device()
    if device == "cuda":
        return {"torch_dtype": torch.float16, "device_map": "auto"}
    return {"torch_dtype": torch.float32}


def run_variant(vid: str, retriever_model: str, lr: float):
    print(f"\n{'=' * 70}")
    print(f"Variant {vid}: retriever={retriever_model.split('/')[-1]}, lr={lr}")
    print(f"{'=' * 70}")

    AcceleratorState._reset_state()

    # Load data
    data = setup_dataset(
        "nfcorpus",
        max_train=MAX_TRAIN,
        max_server_val=MAX_SERVER_VAL,
        max_final_test=MAX_FINAL_TEST,
        max_shared_quality=MAX_SHARED_QUALITY,
        max_docs=MAX_DOCS,
        seed=SEED,
        retriever_model=retriever_model,
    )

    retriever = data["retriever"]
    knowledge_store = data["knowledge_store"]
    train_pairs = data["train_pairs"]
    server_val_pairs = data["server_val_pairs"]
    final_test_pairs = data["final_test_pairs"]

    # Pre-training evaluation
    pre_val = evaluate_retriever(retriever, knowledge_store, server_val_pairs, top_k=TOP_K)
    pre_test = evaluate_retriever(retriever, knowledge_store, final_test_pairs, top_k=TOP_K)

    print(f"  PRE-TRAIN  val  MRR={pre_val['mrr']:.6f}  NDCG={pre_val['ndcg_at_k']:.6f}  Recall={pre_val['recall_at_k']:.4f}")
    print(f"  PRE-TRAIN  test MRR={pre_test['mrr']:.6f}  NDCG={pre_test['ndcg_at_k']:.6f}  Recall={pre_test['recall_at_k']:.4f}")

    # Build RAG system
    generator = HFPretrainedModelGenerator(
        model_name=GENERATOR_MODEL,
        generation_config=GenerationConfig(
            max_new_tokens=30, do_sample=False, pad_token_id=50256,
        ),
        load_model_kwargs=get_generator_load_kwargs(),
    )

    rag_system = RAGSystem(
        knowledge_store=knowledge_store,
        generator=generator,
        retriever=retriever,
        rag_config=RAGConfig(top_k=TOP_K),
    )

    train_dataset = Dataset.from_dict({
        "query": [p["query"] for p in train_pairs],
        "response": [p["response"] for p in train_pairs],
    })

    tmp_dir = os.path.join(OUTPUT_DIR, f"tmp_{vid}")
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

    retriever_trainer = HuggingFaceTrainerForLSR(
        rag_system=rag_system,
        train_dataset=train_dataset,
        training_arguments=training_args,
    )

    # Train
    print(f"  Training on {len(train_dataset)} clean pairs, lr={lr}, 1 epoch ...")
    AcceleratorState._reset_state()
    result = retriever_trainer.train()
    print(f"  Training loss: {result.loss:.6f}")

    # Post-training evaluation
    post_val = evaluate_retriever(retriever, knowledge_store, server_val_pairs, top_k=TOP_K)
    post_test = evaluate_retriever(retriever, knowledge_store, final_test_pairs, top_k=TOP_K)

    print(f"  POST-TRAIN val  MRR={post_val['mrr']:.6f}  NDCG={post_val['ndcg_at_k']:.6f}  Recall={post_val['recall_at_k']:.4f}")
    print(f"  POST-TRAIN test MRR={post_test['mrr']:.6f}  NDCG={post_test['ndcg_at_k']:.6f}  Recall={post_test['recall_at_k']:.4f}")

    val_delta = (post_val['mrr'] - pre_val['mrr']) / max(pre_val['mrr'], 1e-9) * 100
    test_delta = (post_test['mrr'] - pre_test['mrr']) / max(pre_test['mrr'], 1e-9) * 100

    verdict = "IMPROVED" if test_delta > 5 else ("MARGINAL" if test_delta > 0 else "DEGRADED")
    print(f"  delta val_MRR={val_delta:+.1f}%  delta test_MRR={test_delta:+.1f}%  -> {verdict}")

    return {
        "id": vid,
        "retriever": retriever_model,
        "lr": lr,
        "pre_val_mrr": pre_val["mrr"],
        "pre_test_mrr": pre_test["mrr"],
        "post_val_mrr": post_val["mrr"],
        "post_test_mrr": post_test["mrr"],
        "pre_val_ndcg": pre_val["ndcg_at_k"],
        "post_val_ndcg": post_val["ndcg_at_k"],
        "pre_test_ndcg": pre_test["ndcg_at_k"],
        "post_test_ndcg": post_test["ndcg_at_k"],
        "pre_test_recall": pre_test["recall_at_k"],
        "post_test_recall": post_test["recall_at_k"],
        "test_mrr_delta_pct": test_delta,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 1 Diagnostic: Non-federated LSR training")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(ALL_VARIANTS.keys()),
        help="Comma-separated variant IDs to run (default: all)",
    )
    args = parser.parse_args()

    selected_ids = [v.strip() for v in args.variants.split(",")]
    variants = [(vid, ALL_VARIANTS[vid]) for vid in selected_ids if vid in ALL_VARIANTS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for vid, cfg in variants:
        try:
            r = run_variant(vid, cfg["retriever"], cfg["lr"])
            results.append(r)
        except Exception as e:
            print(f"  Variant {vid} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'=' * 95}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 95}")
    print(f"{'ID':<4} {'Retriever':<42} {'LR':<8} {'Pre MRR':>9} {'Post MRR':>9} {'delta%':>7} {'Verdict'}")
    print("-" * 95)
    for r in results:
        short = r["retriever"].split("/")[-1]
        print(
            f"{r['id']:<4} {short:<42} {r['lr']:<8.0e} "
            f"{r['pre_test_mrr']:>9.6f} {r['post_test_mrr']:>9.6f} "
            f"{r['test_mrr_delta_pct']:>+6.1f}% {r['verdict']}"
        )

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "diagnostic_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    # Gate check
    best = max(results, key=lambda r: r["test_mrr_delta_pct"]) if results else None
    if best and best["test_mrr_delta_pct"] > 5:
        print(f"\nGATE PASSED — Best: {best['id']} ({best['retriever'].split('/')[-1]}, lr={best['lr']:.0e}), +{best['test_mrr_delta_pct']:.1f}%")
        print(f"Proceed to Step 2 with this config.")
    elif best and best["test_mrr_delta_pct"] > 0:
        print(f"\nGATE MARGINAL — Best: {best['id']} shows +{best['test_mrr_delta_pct']:.1f}% (below 5% threshold)")
        print(f"Consider using it anyway or trying more LR values.")
    else:
        print(f"\nGATE FAILED — No variant improved retrieval. LSR training itself may be the problem.")


if __name__ == "__main__":
    main()
