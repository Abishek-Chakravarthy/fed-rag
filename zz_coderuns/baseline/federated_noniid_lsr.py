"""
Federated LSR Training — Non-IID Clients (Flower Simulation)
==============================================================
Client 0 & 1: NFCorpus (Medical) — different subsets
Client 2:     SciFact (Science) — completely different domain

This demonstrates the "domain heterogeneity" problem:
the Science client's updates may degrade the medical retriever.

Metrics collected per round:
  - Per-client training loss
  - Aggregated loss

Results saved to: federated_noniid_results.csv
"""

import torch
torch.set_num_threads(1)

import os
import csv
import hashlib
import logging
import random
import numpy as np
from typing import Tuple

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import GenerationConfig
from accelerate.state import AcceleratorState

from fed_rag import RAGSystem, RAGConfig
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.trainers import HuggingFaceTrainerForLSR
from fed_rag.trainer_managers import HuggingFaceRAGTrainerManager

import flwr as fl
from flwr.common import Metrics
from flwr.common.parameter import ndarrays_to_parameters
from fed_rag.fl_tasks.huggingface import _get_weights

from prepare_beir_data import (
    load_beir_dataset,
    build_train_eval_pairs,
    build_knowledge_store,
    create_retriever,
    evaluate_retriever,
    filter_eval_pairs_by_store,
    TOP_K,
    SEED,
)


# ============================================================
# CONFIG
# ============================================================

NUM_ROUNDS = 3
NUM_CLIENTS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-6
GENERATOR_MODEL = "distilgpt2"
MAX_TRAIN_PER_CLIENT = 160  # training pairs per client
MAX_EVAL = 100               # held out for evaluation (from NFCorpus)
MAX_DOCS = 1000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("federated_noniid_lsr")


# ============================================================
# GLOBAL STATE
# ============================================================

CLIENT_TRAIN_DATA = {}      # cid → list of train pairs
CLIENT_DOMAINS = {}         # cid → domain name
KNOWLEDGE_STORE = None      # shared knowledge store (NFCorpus for eval)
EVAL_PAIRS = []
DOC_LOOKUP = {}
ROUND_METRICS = []
_current_round = [0]


# ============================================================
# NON-IID DATA SETUP
# ============================================================

def setup_noniid_data():
    """
    Create heterogeneous client data:
      Client 0: NFCorpus subset A (Medical)
      Client 1: NFCorpus subset B (Medical, different queries)
      Client 2: SciFact (Science — completely different domain!)
    """
    global DOC_LOOKUP

    # ── Load NFCorpus ──
    nf_doc_lookup, nf_query_lookup, nf_qrels = load_beir_dataset("nfcorpus")
    nf_train_pairs, nf_eval_pairs = build_train_eval_pairs(
        nf_doc_lookup, nf_query_lookup, nf_qrels,
        max_train=MAX_TRAIN_PER_CLIENT * 2,  # split between client 0 and 1
        max_eval=MAX_EVAL * 5,  # request extra, most will be filtered out
    )

    # ── Load SciFact ──
    sf_doc_lookup, sf_query_lookup, sf_qrels = load_beir_dataset("scifact")
    sf_train_pairs, _ = build_train_eval_pairs(
        sf_doc_lookup, sf_query_lookup, sf_qrels,
        max_train=MAX_TRAIN_PER_CLIENT,
        max_eval=0,
    )

    # ── Split NFCorpus between client 0 and 1 ──
    midpoint = len(nf_train_pairs) // 2
    client_0_data = nf_train_pairs[:midpoint]
    client_1_data = nf_train_pairs[midpoint:]

    # ── Client 2 gets SciFact ──
    client_2_data = sf_train_pairs[:MAX_TRAIN_PER_CLIENT]

    splits = {
        "0": client_0_data,
        "1": client_1_data,
        "2": client_2_data,
    }
    domains = {
        "0": "NFCorpus (Medical-A)",
        "1": "NFCorpus (Medical-B)",
        "2": "SciFact (Science)",
    }

    print(f"\n🔀 Non-IID Client Data Distribution:")
    for cid, data in splits.items():
        print(f"  Client {cid} [{domains[cid]}]: {len(data)} train pairs")

    # Use NFCorpus as the primary knowledge store (evaluation domain)
    DOC_LOOKUP = nf_doc_lookup

    return splits, domains, nf_eval_pairs, nf_doc_lookup


# ============================================================
# UTILITIES
# ============================================================

def hash_weights(model):
    flat = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    return hashlib.md5(flat.tobytes()).hexdigest()


# ============================================================
# CLIENT FACTORY
# ============================================================

def client_fn(cid: str):
    """Create a Flower client for the given client ID."""
    AcceleratorState._reset_state()

    domain = CLIENT_DOMAINS.get(cid, "unknown")
    logger.info(f"Client {cid} [{domain}]: Creating RAG system...")

    retriever = create_retriever()
    generator = HFPretrainedModelGenerator(
        model_name=GENERATOR_MODEL,
        generation_config=GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=50256,
        ),
        load_model_kwargs={"torch_dtype": torch.float32},
    )

    rag_system = RAGSystem(
        knowledge_store=KNOWLEDGE_STORE,
        generator=generator,
        retriever=retriever,
        rag_config=RAGConfig(top_k=TOP_K),
    )

    # Get this client's data
    data = CLIENT_TRAIN_DATA[cid]
    train_dataset = Dataset.from_dict({
        "query": [p["query"] for p in data],
        "response": [p["response"] for p in data],
    })

    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"fl_noniid_output/client_{cid}"),
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

    model = retriever_trainer.model

    # Use fed-rag FL task to create Flower client
    fl_task = manager.get_federated_task()
    flower_client = fl_task.client(
        model=model,
        train_dataset=train_dataset,
        val_dataset=train_dataset,
    )

    logger.info(f"Client {cid} [{domain}]: Ready ({len(train_dataset)} examples)")
    return flower_client.to_client()


# ============================================================
# METRICS AGGREGATION
# ============================================================

def weighted_average(metrics: list[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training losses and log per-client details."""
    _current_round[0] += 1
    current_round = _current_round[0]

    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    avg_loss = sum(losses) / sum(examples)

    print(f"\n  ⚙️  Round {current_round}: Aggregated loss = {avg_loss:.6f}")
    for i, (n, m) in enumerate(metrics):
        domain = CLIENT_DOMAINS.get(str(i), "?")
        print(f"      Client {i} [{domain}]: loss={m['loss']:.6f}, n={n}")

    ROUND_METRICS.append({
        "round": current_round,
        "avg_loss": avg_loss,
        "client_losses": [m["loss"] for _, m in metrics],
        "client_sizes": list(examples),
    })

    return {"loss": avg_loss}


# ============================================================
# MAIN
# ============================================================

def main():
    global CLIENT_TRAIN_DATA, CLIENT_DOMAINS, KNOWLEDGE_STORE, EVAL_PAIRS

    print("=" * 70)
    print("FEDERATED LSR TRAINING — NON-IID CLIENTS (Flower Simulation)")
    print(f"  Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS}")
    print(f"  Client 0,1: NFCorpus (Medical) | Client 2: SciFact (Science)")
    print("=" * 70)

    # ── Setup non-IID data ──
    splits, domains, eval_pairs, doc_lookup = setup_noniid_data()
    CLIENT_TRAIN_DATA = splits
    CLIENT_DOMAINS = domains
    EVAL_PAIRS = eval_pairs

    # ── Build knowledge store from NFCorpus (evaluation domain) ──
    retriever = create_retriever()
    KNOWLEDGE_STORE = build_knowledge_store(doc_lookup, retriever, MAX_DOCS)

    # ── Filter eval pairs to only include docs in the knowledge store ──
    EVAL_PAIRS = filter_eval_pairs_by_store(EVAL_PAIRS, KNOWLEDGE_STORE)

    # ── Pre-training evaluation ──
    pre_metrics = evaluate_retriever(retriever, KNOWLEDGE_STORE, EVAL_PAIRS, top_k=TOP_K)
    print(f"\n📈 Pre-training metrics (evaluated on NFCorpus/Medical):")
    for k, v in pre_metrics.items():
        print(f"    {k}: {v:.4f}")

    # ── Get initial parameters ──
    AcceleratorState._reset_state()
    model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
    initial_hash = hash_weights(model)
    ndarrays = _get_weights(model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    print(f"\n🔎 Initial global model hash: {initial_hash[:16]}")

    # ── FedAvg strategy ──
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )

    # ── Run simulation ──
    print(f"\n🚀 Starting Flower simulation (Non-IID)...")
    print(f"{'─' * 70}\n")

    _current_round[0] = 0
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    # ── Save results to CSV ──
    csv_path = os.path.join(OUTPUT_DIR, "federated_noniid_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "round", "avg_loss",
            "client_0_loss", "client_0_domain",
            "client_1_loss", "client_1_domain",
            "client_2_loss", "client_2_domain",
            "pre_mrr", "pre_recall_at_k", "pre_ndcg_at_k",
        ])
        writer.writeheader()

        for rm in ROUND_METRICS:
            row = {
                "round": rm["round"],
                "avg_loss": f"{rm['avg_loss']:.6f}",
            }
            for i, cl in enumerate(rm["client_losses"]):
                row[f"client_{i}_loss"] = f"{cl:.6f}"
                row[f"client_{i}_domain"] = CLIENT_DOMAINS.get(str(i), "?")
            row["pre_mrr"] = f"{pre_metrics['mrr']:.6f}"
            row["pre_recall_at_k"] = f"{pre_metrics['recall_at_k']:.6f}"
            row["pre_ndcg_at_k"] = f"{pre_metrics['ndcg_at_k']:.6f}"
            writer.writerow(row)

    print(f"\n{'=' * 70}")
    print(f"✅ FEDERATED NON-IID TRAINING COMPLETE")
    print(f"📄 Results saved to: {csv_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
