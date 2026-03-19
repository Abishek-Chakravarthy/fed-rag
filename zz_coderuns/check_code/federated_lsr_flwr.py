"""
Federated LSR Training — Using Flower Simulation Infrastructure
================================================================
Uses fl.simulation.start_simulation() with the fed-rag FL task API.
Each client gets its own RAG system, trainer, and disjoint data split.
Flower handles weight broadcast, local training, and FedAvg aggregation.

Requires: pip install "flwr[simulation]"
"""

import torch
torch.set_num_threads(1)

import hashlib
import logging
import numpy as np
from typing import Tuple

from datasets import Dataset
from fed_rag import RAGSystem, RAGConfig
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.trainers import HuggingFaceTrainerForLSR
from fed_rag.trainer_managers import HuggingFaceRAGTrainerManager
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import GenerationConfig
from prepare_knowledge import setup_knowledge_store
from accelerate.state import AcceleratorState

import flwr as fl
from flwr.common import Metrics
from flwr.common.parameter import ndarrays_to_parameters
from fed_rag.fl_tasks.huggingface import _get_weights

# ================================
# LOGGING
# ================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger("federated_lsr_flwr")

# ================================
# CONFIG
# ================================

NUM_ROUNDS = 3
NUM_CLIENTS = 2

# ================================
# DATA — disjoint splits
# ================================

CLIENT_DATA = {
    "0": [
        {"query": "Where is the Eiffel Tower?", "response": "Paris, France"},
        {"query": "What is the capital of Japan?", "response": "Tokyo"},
        {"query": "At what temperature does water boil?", "response": "100 degrees Celsius"},
        {"query": "What does the Earth orbit?", "response": "The Sun"},
        {"query": "What is Python?", "response": "A programming language"},
    ],
    "1": [
        {"query": "Where is the Great Wall?", "response": "China"},
        {"query": "What is the highest mountain?", "response": "Mount Everest"},
        {"query": "What is the largest ocean?", "response": "Pacific Ocean"},
        {"query": "What is the capital of UK?", "response": "London"},
        {"query": "What is the speed of light?", "response": "299,792 km/s"},
    ],
}


# ================================
# UTILITIES
# ================================

def hash_weights(model):
    state_dict = model.state_dict()
    flat = np.concatenate([v.detach().cpu().numpy().flatten() for v in state_dict.values()])
    return hashlib.md5(flat.tobytes()).hexdigest()


# ================================
# CLIENT FACTORY (called by Flower per client per round)
# ================================

def client_fn(cid: str):
    """Create a Flower client for the given client ID.

    Each call creates a fresh RAG system + trainer to simulate
    separate client machines. Flower handles weight synchronization.
    """
    AcceleratorState._reset_state()

    logger.info(f"Client {cid}: Creating RAG system and trainer...")

    knowledge_store, retriever = setup_knowledge_store()

    generator = HFPretrainedModelGenerator(
        model_name="sshleifer/tiny-gpt2",
        generation_config=GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=50256,
        ),
        load_model_kwargs={"torch_dtype": torch.float32},
    )

    rag_system = RAGSystem(
        knowledge_store=knowledge_store,
        generator=generator,
        retriever=retriever,
        rag_config=RAGConfig(top_k=2),
    )

    data = CLIENT_DATA[cid]
    train_dataset = Dataset.from_dict({
        "query": [ex["query"] for ex in data],
        "response": [ex["response"] for ex in data],
    })

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"./fl_output/client_{cid}",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        learning_rate=1e-4,
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

    # Use the fed-rag FL task infrastructure
    fl_task = manager.get_federated_task()

    # Build a Flower client — this wraps the training loop
    flower_client = fl_task.client(
        model=model,
        train_dataset=train_dataset,
        val_dataset=train_dataset,  # using train as val since this is a demo
    )

    logger.info(f"Client {cid}: Ready (hash: {hash_weights(model)[:12]}...)")

    return flower_client.to_client()


# ================================
# AGGREGATION METRICS
# ================================

def weighted_average(metrics: list[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training loss across clients (weighted by dataset size)."""
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    avg_loss = sum(losses) / sum(examples)

    print(f"\n  ⚙️  Server: Aggregated loss = {avg_loss:.6f}")
    return {"loss": avg_loss}


# ================================
# MAIN
# ================================

def main():
    print("=" * 60)
    print("FEDERATED LSR TRAINING — Flower Simulation")
    print(f"  Clients: {NUM_CLIENTS}  |  Rounds: {NUM_ROUNDS}")
    print("=" * 60)

    # Get initial model hash for comparison
    AcceleratorState._reset_state()
    _, retriever = setup_knowledge_store()
    global_model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
    initial_hash = hash_weights(global_model)
    print(f"\n🔎 Initial global model hash: {initial_hash}")

    # Get initial model weights to provide to FedAvg strategy
    # (avoids Flower requesting params from a Ray worker, which can cause serialization issues)
    ndarrays = _get_weights(global_model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    # FedAvg strategy with initial parameters
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # all clients train every round
        fraction_evaluate=0.0,      # skip evaluation (placeholder anyway)
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )

    print(f"\n🚀 Starting Flower simulation...")
    print(f"{'─' * 60}\n")

    # Run Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    print(f"\n{'=' * 60}")
    print("✅ FLOWER FEDERATED TRAINING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
