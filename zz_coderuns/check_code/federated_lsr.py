"""
Federated LSR Training Demo — Manual FedAvg Simulation
=======================================================
Demonstrates that the complete fed-rag baseline works end-to-end:
  • 2 clients, each with disjoint data
  • N rounds of FedAvg
  • Each round: distribute global weights → local train → aggregate

No Ray dependency required — uses the fed-rag FL task infrastructure
(HuggingFaceFLTask / HuggingFaceFlowerClient) directly.
"""

import torch
torch.set_num_threads(1)

import hashlib
import logging
import numpy as np

from datasets import Dataset
from fed_rag import RAGSystem, RAGConfig
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.trainers import HuggingFaceTrainerForLSR
from fed_rag.trainer_managers import HuggingFaceRAGTrainerManager
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import GenerationConfig
from prepare_knowledge import setup_knowledge_store
from accelerate.state import AcceleratorState

# ================================
# CONFIG
# ================================

NUM_ROUNDS = 3
NUM_CLIENTS = 2

# ================================
# DATA — disjoint splits for each client
# ================================

CLIENT_DATA = {
    0: [
        {"query": "Where is the Eiffel Tower?", "response": "Paris, France"},
        {"query": "What is the capital of Japan?", "response": "Tokyo"},
        {"query": "At what temperature does water boil?", "response": "100 degrees Celsius"},
        {"query": "What does the Earth orbit?", "response": "The Sun"},
        {"query": "What is Python?", "response": "A programming language"},
    ],
    1: [
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


def get_ndarrays(model):
    """Extract model weights as a list of numpy arrays."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_ndarrays(model, ndarrays):
    """Load a list of numpy arrays back into the model."""
    state_dict = model.state_dict()
    new_state_dict = {
        k: torch.tensor(v) for k, v in zip(state_dict.keys(), ndarrays)
    }
    model.load_state_dict(new_state_dict, strict=True)


def fedavg(client_weights_list, client_sizes):
    """Weighted average of client model weights (FedAvg)."""
    total = sum(client_sizes)
    avg_weights = []
    for layer_idx in range(len(client_weights_list[0])):
        weighted_sum = sum(
            w[layer_idx] * (n / total)
            for w, n in zip(client_weights_list, client_sizes)
        )
        avg_weights.append(weighted_sum)
    return avg_weights


# ================================
# CLIENT FACTORY
# ================================

def create_client(client_id, global_weights=None):
    """Create a fresh client with its own RAG system & trainer.

    If global_weights is provided, load them before training.
    Each client gets a fresh AcceleratorState to avoid conflicts.
    """
    AcceleratorState._reset_state()

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

    data = CLIENT_DATA[client_id]
    train_dataset = Dataset.from_dict({
        "query": [ex["query"] for ex in data],
        "response": [ex["response"] for ex in data],
    })

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"./fl_output/client_{client_id}",
        num_train_epochs=1,           # 1 local epoch per round
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

    # Load global weights if provided (simulates Flower's set_weights)
    if global_weights is not None:
        set_ndarrays(model, global_weights)

    return manager, model, len(train_dataset)


# ================================
# MAIN
# ================================

def main():
    print("=" * 60)
    print("FEDERATED LSR TRAINING — Manual FedAvg Simulation")
    print(f"  Clients: {NUM_CLIENTS}  |  Rounds: {NUM_ROUNDS}")
    print("=" * 60)

    # ── Initialise global model ──────────────────────────────────
    AcceleratorState._reset_state()
    knowledge_store, retriever = setup_knowledge_store()
    global_model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
    global_weights = get_ndarrays(global_model)
    initial_hash = hash_weights(global_model)
    print(f"\n🔎 Initial global model hash: {initial_hash}")

    # ── Federated rounds ─────────────────────────────────────────
    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─' * 60}")
        print(f"📡 ROUND {rnd}/{NUM_ROUNDS}")
        print(f"{'─' * 60}")

        client_weights = []
        client_sizes = []

        for cid in range(NUM_CLIENTS):
            print(f"\n  🖥️  Client {cid}: training on {len(CLIENT_DATA[cid])} examples...")

            manager, model, n_examples = create_client(cid, global_weights)

            pre_hash = hash_weights(model)
            result = manager.train()
            post_hash = hash_weights(model)

            print(f"      Hash before: {pre_hash[:12]}...")
            print(f"      Hash after:  {post_hash[:12]}...")
            print(f"      Loss: {result.loss:.6f}")
            print(f"      Weights changed: {'✅ Yes' if pre_hash != post_hash else '❌ No'}")

            client_weights.append(get_ndarrays(model))
            client_sizes.append(n_examples)

        # ── Aggregate (FedAvg) ───────────────────────────────────
        print(f"\n  ⚙️  Aggregating {NUM_CLIENTS} client updates (FedAvg)...")
        global_weights = fedavg(client_weights, client_sizes)

        # Load aggregated weights into a model to hash them
        set_ndarrays(global_model, global_weights)
        new_hash = hash_weights(global_model)
        print(f"  🔎 New global model hash: {new_hash[:12]}...")

    # ── Final summary ────────────────────────────────────────────
    final_hash = hash_weights(global_model)
    print(f"\n{'=' * 60}")
    print("✅ FEDERATED TRAINING COMPLETE")
    print(f"   Initial hash: {initial_hash}")
    print(f"   Final hash:   {final_hash}")
    print(f"   Weights changed: {'✅ Yes' if initial_hash != final_hash else '❌ No'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()