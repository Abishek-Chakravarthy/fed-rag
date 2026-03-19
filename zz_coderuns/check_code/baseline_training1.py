"""
Centralized Retriever (LSR) Training Verification Script
Verifies gradient flow and weight updates
"""

import torch
torch.set_num_threads(1)

import hashlib
import numpy as np
import logging
import os

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainingArguments

from fed_rag.core.rag_system import RAGSystem
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.data_structures import KnowledgeNode, NodeType, RAGConfig
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.generators.huggingface import HFPretrainedModelGenerator
from fed_rag.trainers.huggingface.lsr import HuggingFaceTrainerForLSR
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager
from accelerate.state import AcceleratorState


# ===============================
# Logging
# ===============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ===============================
# Gradient + Weight Utilities
# ===============================

def flatten_weights(model):
    return np.concatenate(
        [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    )

def hash_weights(model):
    flat = flatten_weights(model)
    return hashlib.md5(flat.tobytes()).hexdigest()

def inspect_gradients(model):
    total = 0
    nonzero = 0
    total_norm = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total += 1
            if param.grad is not None:
                norm = param.grad.norm().item()
                if norm > 0:
                    nonzero += 1
                total_norm += norm

    print("\n🔍 RETRIEVER GRADIENT CHECK")
    print(f"Trainable params: {total}")
    print(f"Params with non-zero grad: {nonzero}")
    print(f"Total grad norm: {total_norm:.6f}")

    if nonzero == 0:
        print("❌ No gradients flowed.")
    else:
        print("✅ Gradients flowed correctly.")


# ===============================
# Knowledge Store
# ===============================

def create_knowledge_store():
    retriever = HFSentenceTransformerRetriever(
        query_model_name="sentence-transformers/all-MiniLM-L6-v2",
        context_model_name="sentence-transformers/all-MiniLM-L6-v2",
        load_model_at_init=True,
    )

    knowledge_texts = [
        "Machine learning enables systems to learn from experience.",
        "Deep learning uses multi-layer neural networks.",
        "Natural language processing enables human-computer interaction.",
        "Computer vision processes visual data.",
        "Artificial intelligence simulates human reasoning.",
        "Neural networks consist of interconnected neurons.",
        "Supervised learning uses labeled data.",
        "Unsupervised learning finds hidden patterns.",
    ]

    store = InMemoryKnowledgeStore()
    nodes = []

    for i, text in enumerate(knowledge_texts):
        embedding = retriever.encode_context(text)
        embedding = embedding.flatten().tolist()

        node = KnowledgeNode(
            node_id=str(i),
            embedding=embedding,
            node_type=NodeType.TEXT,
            text_content=text,
        )
        nodes.append(node)

    store.load_nodes(nodes)
    return store, retriever


# ===============================
# Dataset
# ===============================

def create_dataset():
    return Dataset.from_dict(
        {
            "query": [
                "What is machine learning?",
                "Explain deep learning",
                "What is NLP?",
                "Describe AI",
            ],
            "response": [
                "Machine learning enables systems to learn from data.",
                "Deep learning uses neural networks with multiple layers.",
                "NLP stands for Natural Language Processing.",
                "AI simulates human intelligence.",
            ],
        }
    )


# ===============================
# Main Verification Logic
# ===============================

def main():

    AcceleratorState._reset_state()

    print("=" * 70)
    print("CENTRALIZED RETRIEVER (LSR) TRAINING VERIFICATION")
    print("=" * 70)

    # Knowledge store + retriever
    store, retriever = create_knowledge_store()

    # Dummy generator (required for RAGSystem)
    generator = HFPretrainedModelGenerator(
        model_name="gpt2",
        load_model_kwargs={"device_map": "cpu"},
    )

    rag_system = RAGSystem(
        retriever=retriever,
        generator=generator,
        knowledge_store=store,
        rag_config=RAGConfig(top_k=2),
    )

    dataset = create_dataset()

    # Training args
    training_args = SentenceTransformerTrainingArguments(
        output_dir="./retriever_checkpoint",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        learning_rate=5e-5,
    )

    retriever_trainer = HuggingFaceTrainerForLSR(
        rag_system=rag_system,
        train_dataset=dataset,
        training_arguments=training_args,
    )

    manager = HuggingFaceRAGTrainerManager(
        rag_system=rag_system,
        retriever_trainer=retriever_trainer,
        mode="retriever",
    )

    # Get retriever model
    model = (
        rag_system.retriever.encoder
        if rag_system.retriever.encoder
        else rag_system.retriever.query_encoder
    )

    print("\n🔎 HASH BEFORE TRAINING:", hash_weights(model))
    before_flat = flatten_weights(model)

    # Manual gradient check
    hf_trainer = retriever_trainer.hf_trainer_obj
    hf_trainer.model.train()

    dataloader = hf_trainer.get_train_dataloader()
    optimizer = hf_trainer.create_optimizer()

    batch = next(iter(dataloader))
    loss = hf_trainer.compute_loss(hf_trainer.model, batch)

    print(f"\n📉 Initial LSR Loss: {loss.item():.6f}")

    loss.backward()

    inspect_gradients(model)

    optimizer.step()
    optimizer.zero_grad()

    # Full training
    result = manager.train()

    print("\n🔎 HASH AFTER TRAINING:", hash_weights(model))
    after_flat = flatten_weights(model)

    update_norm = np.linalg.norm(after_flat - before_flat)
    print("📏 Weight Update L2 Norm:", update_norm)

    if update_norm == 0:
        print("❌ Retriever weights did NOT change.")
    else:
        print("✅ Retriever weights updated successfully.")

    print("\nFinal Training Loss:", result.loss)


if __name__ == "__main__":
    main()