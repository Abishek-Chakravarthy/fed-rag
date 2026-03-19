"""
Centralized LSR Training on BEIR NFCorpus
==========================================
Trains the retriever (LSR) on ALL data pooled together.
This is the "upper bound" baseline — no federation, no data split.

Metrics collected per epoch:
  - Training loss
  - MRR, Recall@k, NDCG@k on held-out evaluation set

Results saved to: centralized_results.csv
"""

import torch
torch.set_num_threads(1)

import os
import csv
import hashlib
import logging
import numpy as np

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import GenerationConfig
from accelerate.state import AcceleratorState

from fed_rag import RAGSystem, RAGConfig
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.trainers import HuggingFaceTrainerForLSR
from fed_rag.trainer_managers import HuggingFaceRAGTrainerManager

from prepare_beir_data import (
    setup_dataset,
    evaluate_retriever,
    create_retriever,
    build_knowledge_store,
    TOP_K,
)

# ============================================================
# CONFIG
# ============================================================

NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-6
GENERATOR_MODEL = "distilgpt2"
DATASET_NAME = "nfcorpus"
MAX_TRAIN = 250
MAX_EVAL = 100
MAX_DOCS = 1000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("centralized_lsr")


# ============================================================
# UTILITIES
# ============================================================

def hash_weights(model):
    flat = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    return hashlib.md5(flat.tobytes()).hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CENTRALIZED LSR TRAINING — BEIR NFCorpus")
    print(f"  Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print("=" * 70)

    # ── Load data ──
    data = setup_dataset(DATASET_NAME, max_train=MAX_TRAIN, max_eval=MAX_EVAL, max_docs=MAX_DOCS)
    retriever = data["retriever"]
    knowledge_store = data["knowledge_store"]
    train_dataset = data["train_dataset"]
    eval_pairs = data["eval_pairs"]

    print(f"\n📊 Train size: {len(train_dataset)} | Eval size: {len(eval_pairs)}")

    # ── Create generator (frozen, used as teacher) ──
    generator = HFPretrainedModelGenerator(
        model_name=GENERATOR_MODEL,
        generation_config=GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=50256,
        ),
        load_model_kwargs={"torch_dtype": torch.float32},
    )

    # ── Evaluate BEFORE training ──
    pre_metrics = evaluate_retriever(retriever, knowledge_store, eval_pairs, top_k=TOP_K)
    print(f"\n📈 Pre-training metrics:")
    for k, v in pre_metrics.items():
        print(f"    {k}: {v:.4f}")

    # ── CSV setup ──
    csv_path = os.path.join(OUTPUT_DIR, "centralized_results.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["epoch", "loss", "mrr", "recall_at_k", "ndcg_at_k", "weight_hash"])
    writer.writeheader()

    # Write pre-training row
    model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
    writer.writerow({
        "epoch": 0,
        "loss": "",
        "mrr": f"{pre_metrics['mrr']:.6f}",
        "recall_at_k": f"{pre_metrics['recall_at_k']:.6f}",
        "ndcg_at_k": f"{pre_metrics['ndcg_at_k']:.6f}",
        "weight_hash": hash_weights(model)[:16],
    })
    csv_file.flush()

    # ── Train epoch-by-epoch ──
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'─' * 70}")
        print(f"📘 EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'─' * 70}")

        AcceleratorState._reset_state()

        # Rebuild RAG system (required for fresh trainer per epoch)
        rag_system = RAGSystem(
            retriever=retriever,
            generator=generator,
            knowledge_store=knowledge_store,
            rag_config=RAGConfig(top_k=TOP_K),
        )

        training_args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, "centralized_output"),
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.1,
            weight_decay=0.01,
        )

        retriever_trainer = HuggingFaceTrainerForLSR(
            rag_system=rag_system,
            train_dataset=train_dataset,
            training_arguments=training_args,
        )

        manager = HuggingFaceRAGTrainerManager(
            mode="retriever",
            retriever_trainer=retriever_trainer,
        )

        # Train
        pre_hash = hash_weights(model)
        result = manager.train()
        post_hash = hash_weights(model)

        print(f"  📉 Loss: {result.loss:.6f}")
        print(f"  🔑 Hash: {pre_hash[:12]} → {post_hash[:12]}")
        print(f"  ✅ Weights changed: {'Yes' if pre_hash != post_hash else 'No'}")

        # ── Evaluate after this epoch ──
        # Re-build knowledge store with updated encoder for fair evaluation
        knowledge_store = build_knowledge_store(data["doc_lookup"], retriever, MAX_DOCS)
        metrics = evaluate_retriever(retriever, knowledge_store, eval_pairs, top_k=TOP_K)

        print(f"  📈 MRR: {metrics['mrr']:.4f} | Recall@{TOP_K}: {metrics['recall_at_k']:.4f} | NDCG@{TOP_K}: {metrics['ndcg_at_k']:.4f}")

        # Write to CSV
        writer.writerow({
            "epoch": epoch,
            "loss": f"{result.loss:.6f}",
            "mrr": f"{metrics['mrr']:.6f}",
            "recall_at_k": f"{metrics['recall_at_k']:.6f}",
            "ndcg_at_k": f"{metrics['ndcg_at_k']:.6f}",
            "weight_hash": post_hash[:16],
        })
        csv_file.flush()

    csv_file.close()
    print(f"\n{'=' * 70}")
    print(f"✅ CENTRALIZED TRAINING COMPLETE")
    print(f"📄 Results saved to: {csv_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
