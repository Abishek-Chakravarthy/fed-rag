import torch
torch.set_num_threads(1)

import os
import csv
import json
import hashlib
import logging
import random
import argparse
import numpy as np
from typing import Tuple

from datasets import Dataset
from datasets.utils import logging as datasets_logging
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import GenerationConfig
from transformers.utils import logging as transformers_logging
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
    setup_dataset,
    evaluate_retriever,
    create_retriever,
    load_beir_dataset,
    build_train_eval_pairs,
    TOP_K,
    SEED,
)

from quality_aware_fedavg import QualityAwareFedAvg


NUM_ROUNDS = 4
NUM_CLIENTS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-6
GENERATOR_MODEL = "distilgpt2"
DATASET_NAME = "nfcorpus"
MAX_TRAIN = 1500 # Limits the training set to 1500 query-response pairs.
MAX_EVAL = 300 # Limits the held-out evaluation set to 300 query-response pairs.
MAX_DOCS = 3000 # Limits the knowledge store to 3000 documents.
QUALITY_HOLDOUT_RATIO = 0.2 # Fraction of each client's clean data reserved for validation.
NOISE_RATIO = 0.8 # 80% of the noisy client's train split will be corrupted.
NOISE_MODE = "hard_negative" # The type of noise to introduce.
NOISY_CLIENT_ID = "2" # The client to introduce noise to.
CLIENT_SPLIT_MODE = "unequal"
NOISY_CLIENT_DATA_FRACTION = 0.4
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUTPUT_DIR, "output_csv_files")
LOG_DIR = os.path.join(OUTPUT_DIR, "output_log_files")
TMP_TRAINING_DIR = os.path.join(OUTPUT_DIR, ".tmp_training")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # PyTorch tries to use the MPS (Metal Performance Shaders) backend to accelerate work on your GPU.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("federated_noisy_qa")
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
datasets_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


CLIENT_TRAIN_DATA = {} # A dictionary to hold the training data for each client.
CLIENT_QUALITY_HOLDOUTS = {} # Clean per-client holdouts used for quality scoring.
KNOWLEDGE_STORE = None # The knowledge store for the RAG system.
EVAL_PAIRS = [] # The evaluation pairs for the RAG system.
DOC_LOOKUP = {} # A dictionary to look up documents in the knowledge store.
ROUND_METRICS = [] # A list to store the metrics for each round.
ALPHA = 0.5 # The alpha parameter for the QA-FedAvg algorithm.
CURRENT_SEED = SEED
CURRENT_NUM_ROUNDS = NUM_ROUNDS
CURRENT_LOCAL_EPOCHS = 1
CURRENT_NOISE_MODE = NOISE_MODE # A global tracker for how the "bad" client is corrupting its data (e.g., shuffle, cross_domain, etc.)
NOISE_CONTEXT = {} # A global dictionary to hold extra data needed for certain noise modes (e.g., the pool of cross-domain documents).
REFERENCE_RETRIEVER = None # Frozen reference retriever used to construct hard in-domain negatives.


def get_runtime_device() -> str:
    import torch as _torch
    if _torch.cuda.is_available():
        return "cuda"
    if _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_generator_load_kwargs() -> dict:
    import torch as _torch
    device = get_runtime_device()
    if device == "cuda":
        return {"torch_dtype": _torch.float16, "device_map": "auto"}
    return {"torch_dtype": _torch.float32}


def get_client_resources() -> dict[str, float]:
    import torch as _torch
    if _torch.cuda.is_available():
        return {"num_cpus": 2, "num_gpus": 0.34}
    return {"num_cpus": 1, "num_gpus": 0}


def split_iid(train_pairs, num_clients):
    """
    Splits the training data into IID (Independent and Identically Distributed) chunks for each client.

    Args:
        train_pairs (list): List of training query-response pairs.
        num_clients (int): Number of clients to split the data between.

    Returns:
        dict: A mapping from client ID (str) to their allocated list of training pairs.
    """
    random.seed(CURRENT_SEED)
    shuffled = train_pairs.copy()
    random.shuffle(shuffled)

    splits = {}
    chunk_size = len(shuffled) // num_clients
    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(shuffled)
        splits[str(i)] = shuffled[start:end]

    return splits


def split_unequal_noisy(
    train_pairs,
    num_clients,
    noisy_client=NOISY_CLIENT_ID,
    noisy_fraction=NOISY_CLIENT_DATA_FRACTION,
):
    random.seed(CURRENT_SEED)
    shuffled = train_pairs.copy()
    random.shuffle(shuffled)

    noisy_count = int(len(shuffled) * noisy_fraction)
    clean_count = len(shuffled) - noisy_count

    clean_clients = [str(i) for i in range(num_clients) if str(i) != noisy_client]
    clean_chunk = clean_count // len(clean_clients)

    splits = {}
    offset = 0
    for i, cid in enumerate(clean_clients):
        end = offset + clean_chunk if i < len(clean_clients) - 1 else clean_count
        splits[cid] = shuffled[offset:end]
        offset = end

    splits[noisy_client] = shuffled[clean_count:]

    return splits


def reserve_clean_holdouts(
    splits,
    *,
    holdout_ratio: float = QUALITY_HOLDOUT_RATIO,
):
    train_splits = {}
    holdout_splits = {}

    for cid, pairs in sorted(splits.items(), key=lambda item: client_sort_key(item[0])):
        pairs_copy = pairs.copy()
        rng = random.Random(CURRENT_SEED + 1000 + int(cid))
        rng.shuffle(pairs_copy)

        if len(pairs_copy) <= 2:
            holdout_count = 1 if len(pairs_copy) > 1 else 0
        else:
            holdout_count = max(1, int(len(pairs_copy) * holdout_ratio))
            holdout_count = min(holdout_count, len(pairs_copy) - 1)

        holdout_splits[cid] = pairs_copy[:holdout_count]
        train_splits[cid] = pairs_copy[holdout_count:]

    return train_splits, holdout_splits


def split_noisy(
    train_pairs,
    num_clients,
    noise_ratio=NOISE_RATIO,
    noisy_client=NOISY_CLIENT_ID,
    noise_mode=NOISE_MODE,
):
    """
    Partitions data across clients and introduces artificial corruption into one specific client's data.

    Args:
        train_pairs (list): The complete set of training data.
        num_clients (int): Total number of clients in the federated simulation.
        noise_ratio (float): The fraction of data to corrupt in the noisy client (0.0 to 1.0).
        noisy_client (str): The ID of the client that will receive corrupted data.
        noise_mode (str): The method of corruption ('shuffle', 'cross_domain', 'random_negative', or 'mixed').

    Returns:
        tuple[dict, dict]: Training splits and clean validation holdouts.
    """
    if CLIENT_SPLIT_MODE == "equal":
        splits = split_iid(train_pairs, num_clients)
    elif CLIENT_SPLIT_MODE == "unequal":
        splits = split_unequal_noisy(
            train_pairs,
            num_clients,
            noisy_client=noisy_client,
            noisy_fraction=NOISY_CLIENT_DATA_FRACTION,
        )
    else:
        raise ValueError(f"Unsupported client split mode: {CLIENT_SPLIT_MODE}")

    train_splits, holdout_splits = reserve_clean_holdouts(splits)

    for cid in sorted(train_splits.keys()):
        if cid != noisy_client:
            print(
                f"  Client {cid}: {len(train_splits[cid])} train + "
                f"{len(holdout_splits[cid])} holdout (CLEAN ✅)"
            )

    data = train_splits[noisy_client].copy()
    num_corrupt = int(len(data) * noise_ratio)

    rng = random.Random(CURRENT_SEED + 999)
    target_slice = data[:num_corrupt]
    original_responses = [p["response"] for p in target_slice]

    if noise_mode == "shuffle":
        replacement_responses = deranged_shuffle(original_responses, rng)
    elif noise_mode == "hard_negative":
        replacement_responses = sample_hard_negative_responses(
            pairs=target_slice,
            rng=rng,
        )
    elif noise_mode == "cross_domain":
        pool = NOISE_CONTEXT.get("cross_domain_pool", [])
        if not pool:
            raise ValueError("Cross-domain noise requested but no pool is available.")
        replacement_responses = [rng.choice(pool) for _ in range(num_corrupt)]
    elif noise_mode == "random_negative":
        replacement_responses = sample_random_negative_responses(
            count=num_corrupt,
            rng=rng,
            doc_lookup=DOC_LOOKUP,
            original_responses=original_responses,
        )
    elif noise_mode == "mixed":
        pool = NOISE_CONTEXT.get("cross_domain_pool", [])
        shuffled = deranged_shuffle(original_responses, rng)
        random_negatives = sample_random_negative_responses(
            count=num_corrupt,
            rng=rng,
            doc_lookup=DOC_LOOKUP,
            original_responses=original_responses,
        )
        replacement_responses = []
        for idx in range(num_corrupt):
            mode_idx = idx % 3
            if mode_idx == 0:
                replacement_responses.append(shuffled[idx])
            elif mode_idx == 1 and pool:
                replacement_responses.append(rng.choice(pool))
            else:
                replacement_responses.append(random_negatives[idx])
    else:
        raise ValueError(f"Unsupported noise mode: {noise_mode}")

    corrupted_data = []
    for i in range(num_corrupt):
        pair = dict(data[i])
        pair["response"] = replacement_responses[i]
        corrupted_data.append(pair)
    corrupted_data.extend(data[num_corrupt:])

    train_splits[noisy_client] = corrupted_data
    print(
        f"  Client {noisy_client}: {len(data)} train + "
        f"{len(holdout_splits[noisy_client])} holdout — ⚠️  "
        f"{num_corrupt} CORRUPTED ({noise_ratio*100:.0f}% noise, mode={noise_mode})"
    )

    return train_splits, holdout_splits


def hash_weights(model):
    """
    Computes an MD5 hash of the model's parameters to track model state.
    Used for verifying that the model has actually changed after aggregation or training.

    Args:
        model (torch.nn.Module): The model whose weights should be hashed.

    Returns:
        str: A hexadecimal MD5 hash representing the model's current weights.
    """
    flat = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    return hashlib.md5(flat.tobytes()).hexdigest()


def client_sort_key(cid: str):
    """
    Generates a sort key for client IDs to ensure they are processed in a consistent order.
    Handles both numeric strings ("1", "2") and non-numeric client IDs.

    Args:
        cid (str): The client ID string.

    Returns:
        tuple: A sortable key where numeric IDs come first in integer order.
    """
    return (0, int(cid)) if cid.isdigit() else (1, cid)


def hash_jsonable(payload) -> str:
    """
    Creates a deterministic MD5 hash for any JSON-serializable object.
    Used to verify the integrity and consistency of data splits by ensuring 
    the same data results in the same hash regardless of dictionary order.

    Args:
        payload (any): The JSON-serializable object to hash.

    Returns:
        str: A hexadecimal MD5 hash of the serialized object.
    """
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def build_split_summary(splits):
    """
    Creates a summary of how training data is distributed across clients.
    Records the count and a fingerprint (hash) of the data for each client to 
    ensure experiment reproducibility.

    Args:
        splits (dict): A dictionary mapping client IDs to their list of query-response pairs.

    Returns:
        dict: A summary mapping client IDs to metadata (num_pairs, pairs_hash).
    """
    summary = {}
    for cid, pairs in sorted(splits.items(), key=lambda item: client_sort_key(item[0])):
        summary[cid] = {
            "num_pairs": len(pairs),
            "pairs_hash": hash_jsonable(pairs),
        }
    return summary


def build_csv_fieldnames(client_ids):
    base_fields = [
        "round",
        "alpha",
        "seed",
        "noise_mode",
        "noise_ratio",
        "avg_loss",
        "avg_train_loss",
        "aggregated_delta_norm",
        "aggregated_model_hash",
        "pre_mrr",
        "pre_recall_at_k",
        "pre_ndcg_at_k",
        "pre_probe_mrr",
        "pre_probe_recall_at_k",
        "pre_probe_ndcg_at_k",
        "post_mrr",
        "post_recall_at_k",
        "post_ndcg_at_k",
    ]
    for cid in client_ids:
        base_fields.extend(
            [
                f"client_{cid}_loss",
                f"client_{cid}_train_loss",
                f"client_{cid}_num_examples",
                f"client_{cid}_quality_score",
                f"client_{cid}_quality_metric_name",
                f"client_{cid}_quality_metric_value",
                f"client_{cid}_size_weight",
                f"client_{cid}_weight",
                f"client_{cid}_delta_norm",
                f"client_{cid}_probe_size",
                f"client_{cid}_probe_mrr",
                f"client_{cid}_probe_recall_at_k",
                f"client_{cid}_probe_ndcg_at_k",
                f"client_{cid}_loss_source",
                f"client_{cid}_loss_stage",
            ]
        )
    return base_fields

# Creates a unique identifier for a specific experimental run based on its configuration.
# eg: alpha_0.5_seed_42_mode_shuffle_ratio_0.7_r3_e1
def build_run_slug(
    *,
    alpha: float,
    seed: int,
    noise_mode: str,
    noise_ratio: float,
    num_rounds: int,
    local_epochs: int,
) -> str:
    mode = noise_mode.replace("-", "_")
    ratio = f"{noise_ratio:.2f}".replace(".", "p")
    return (
        f"alpha_{alpha:.1f}_seed_{seed}_mode_{mode}_"
        f"ratio_{ratio}_r{num_rounds}_e{local_epochs}"
    )


def set_retriever_weights(retriever, parameters):
    model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
    state_dict = model.state_dict()
    updated_state = {}
    for key, value in zip(state_dict.keys(), parameters):
        updated_state[key] = torch.tensor(value)
    model.load_state_dict(updated_state, strict=True)


def make_post_aggregation_evaluator():
    def evaluator(server_round: int, aggregated_ndarrays):
        del server_round
        retriever = create_retriever()
        set_retriever_weights(retriever, aggregated_ndarrays)
        return evaluate_retriever(
            retriever,
            KNOWLEDGE_STORE,
            EVAL_PAIRS,
            top_k=TOP_K,
        )

    return evaluator


def flatten_holdout_pairs(holdout_splits):
    pairs = []
    for cid in sorted(holdout_splits.keys(), key=client_sort_key):
        pairs.extend(holdout_splits[cid])
    return pairs


def compute_quality_holdout_loss(retriever, cid: str):
    holdout_pairs = CLIENT_QUALITY_HOLDOUTS[cid]
    holdout_metrics = evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        holdout_pairs,
        top_k=TOP_K,
    )
    quality_metric_value = (
        0.5 * holdout_metrics.get("mrr", 0.0)
        + 0.5 * holdout_metrics.get("ndcg_at_k", 0.0)
    )
    quality_loss = float(-np.log(max(quality_metric_value, 1e-8)))
    return quality_loss, quality_metric_value, holdout_metrics


def build_cross_domain_response_pool(
    *,
    dataset_name: str = "scifact",
    max_examples: int = 500,
    seed: int = CURRENT_SEED,
):
    """
    Loads a completely different BEIR dataset to create a "pool" of irrelevant responses.
    Used for the 'cross_domain' noise mode to simulate clients providing out-of-domain garbage.

    Args:
        dataset_name (str): The name of the external dataset to pull from.
        max_examples (int): Maximum number of responses to collect.
        seed (int): Seed for consistency.

    Returns:
        list[str]: A list of response strings from the external domain.
    """
    doc_lookup, query_lookup, qrels_ds = load_beir_dataset(dataset_name)
    train_pairs, _ = build_train_eval_pairs(
        doc_lookup,
        query_lookup,
        qrels_ds,
        max_train=max_examples,
        max_eval=0,
        seed=seed,
    )
    return [pair["response"] for pair in train_pairs]


def sample_random_negative_responses(
    *,
    count: int,
    rng: random.Random,
    doc_lookup: dict,
    original_responses: list[str],
):
    """
    Samples random documents from the corpus to act as incorrect/negative responses.

    Args:
        count (int): Number of negatives to generate.
        rng (random.Random): Random number generator instance.
        doc_lookup (dict): The corpus lookup dictionary.
        original_responses (list[str]): The correct responses (used to ensure we don't pick the same text).

    Returns:
        list[str]: A list of random, incorrect response strings.
    """
    docs = list(doc_lookup.values())
    negatives = []
    for original in original_responses:
        replacement = original
        attempts = 0
        while replacement == original and attempts < 10:
            replacement = rng.choice(docs)[:500]
            attempts += 1
        negatives.append(replacement)
        if len(negatives) >= count:
            break
    return negatives


def sample_hard_negative_responses(
    *,
    pairs: list[dict],
    rng: random.Random,
):
    if REFERENCE_RETRIEVER is None:
        raise ValueError("Hard-negative noise requested before reference retriever was initialized.")

    responses = []
    for pair in pairs:
        query_emb = REFERENCE_RETRIEVER.encode_query(pair["query"])[0].tolist()
        results = KNOWLEDGE_STORE.retrieve(query_emb, top_k=max(TOP_K * 3, 30))
        candidates = []
        for _score, node in results:
            doc_id = str(node.metadata.get("doc_id", ""))
            text = str(node.text_content)[:500]
            if doc_id == str(pair.get("doc_id", "")):
                continue
            if text == pair["response"]:
                continue
            candidates.append(text)
            if len(candidates) >= 5:
                break

        if candidates:
            responses.append(rng.choice(candidates))
        else:
            fallback = sample_random_negative_responses(
                count=1,
                rng=rng,
                doc_lookup=DOC_LOOKUP,
                original_responses=[pair["response"]],
            )[0]
            responses.append(fallback)

    return responses


def deranged_shuffle(values: list[str], rng: random.Random) -> list[str]:
    """
    Shuffles a list such that no element remains in its original position (a derangement).
    This simulates a "swapped labels" scenario where every query gets the wrong answer from within the same set.

    Args:
        values (list[str]): The list of strings to shuffle.
        rng (random.Random): Random number generator instance.

    Returns:
        list[str]: Deranged version of the input list.
    """
    if len(values) <= 1:
        return values.copy()

    shuffled = values.copy()
    for _ in range(20):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(values, shuffled)):
            return shuffled

    return shuffled[1:] + shuffled[:1]


def write_run_manifest(
    manifest_path: str,
    *,
    alpha: float,
    seed: int,
    noise_mode: str,
    noise_ratio: float,
    num_rounds: int,
    local_epochs: int,
    initial_hash: str,
    pre_metrics: dict,
    pre_probe_metrics: dict,
    split_summary: dict,
    csv_path: str,
    log_path: str,
    acceptance_path: str,
):
    manifest = {
        "experiment_name": "qa_fedavg_noisy_client",
        "canonical_results_csv": csv_path,
        "canonical_log_file": log_path,
        "acceptance_report": acceptance_path,
        "alpha": alpha,
        "seed": seed,
        "dataset_name": DATASET_NAME,
        "num_rounds": num_rounds,
        "num_clients": NUM_CLIENTS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "local_epochs": local_epochs,
        "max_train_pairs": MAX_TRAIN,
        "max_eval_pairs": MAX_EVAL,
        "max_corpus_docs": MAX_DOCS,
        "quality_holdout_ratio": QUALITY_HOLDOUT_RATIO,
        "client_split_mode": CLIENT_SPLIT_MODE,
        "top_k": TOP_K,
        "noise_ratio": noise_ratio,
        "noise_mode": noise_mode,
        "noisy_client_id": NOISY_CLIENT_ID,
        "initial_model_hash": initial_hash,
        "pre_metrics": pre_metrics,
        "pre_probe_metrics": pre_probe_metrics,
        "client_split_summary": split_summary,
        "train_split_hash": hash_jsonable(split_summary),
        "csv_schema_version": 3,
        "quality_signal": {
            "strategy_metric_key": "loss",
            "client_metric_source": "client_specific_clean_holdout_retrieval",
            "trainer_return_value": "-log(0.5 * (holdout_mrr + holdout_ndcg_at_k) + eps)",
            "loss_stage": "post_local_training_clean_holdout",
            "objective": "log-scaled clean holdout retrieval loss",
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def evaluate_single_run_acceptance(round_infos, alpha: float):
    report = {
        "alpha_zero_matches_fedavg": {
            "applicable": alpha == 0.0,
            "passed": None,
        },
        "higher_loss_clients_downweighted": {
            "applicable": alpha > 0.0,
            "passed": None,
        },
        "distinct_round_trajectories": {
            "applicable": True,
            "passed": None,
        },
    }

    if alpha == 0.0:
        fedavg_ok = True
        for round_info in round_infos:
            for record in round_info["client_records"]:
                if abs(record["combined_weight"] - record["size_weight"]) > 1e-6:
                    fedavg_ok = False
                    break
        report["alpha_zero_matches_fedavg"]["passed"] = fedavg_ok

    if alpha > 0.0:
        downweighted_rounds = 0
        for round_info in round_infos:
            noisiest = max(
                round_info["client_records"],
                key=lambda record: record["loss"],
            )
            if noisiest["combined_weight"] < noisiest["size_weight"]:
                downweighted_rounds += 1
        report["higher_loss_clients_downweighted"]["passed"] = (
            downweighted_rounds == len(round_infos)
        )
        report["higher_loss_clients_downweighted"]["evidence"] = {
            "downweighted_rounds": downweighted_rounds,
            "total_rounds": len(round_infos),
        }

    unique_hashes = {round_info["aggregated_model_hash"] for round_info in round_infos}
    report["distinct_round_trajectories"]["passed"] = len(unique_hashes) == len(round_infos)
    report["distinct_round_trajectories"]["evidence"] = {
        "unique_model_hashes": len(unique_hashes),
        "rounds": len(round_infos),
    }
    return report


def client_fn(cid: str):
    AcceleratorState._reset_state()

    logger.info(f"Client {cid}: Creating RAG system...")

    retriever = create_retriever()
    generator = HFPretrainedModelGenerator(
        model_name=GENERATOR_MODEL,
        generation_config=GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=50256,
        ),
        load_model_kwargs=get_generator_load_kwargs(),
    )

    rag_system = RAGSystem(
        knowledge_store=KNOWLEDGE_STORE,
        generator=generator,
        retriever=retriever,
        rag_config=RAGConfig(top_k=TOP_K),
    )

    data = CLIENT_TRAIN_DATA[cid]
    train_dataset = Dataset.from_dict({
        "query": [p["query"] for p in data],
        "response": [p["response"] for p in data],
    })

    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(
            TMP_TRAINING_DIR,
            f"seed_{CURRENT_SEED}",
            f"alpha_{ALPHA:.1f}",
            f"client_{cid}",
        ),
        num_train_epochs=CURRENT_LOCAL_EPOCHS,
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

    fl_task = manager.get_federated_task()
    flower_client = fl_task.client(
        model=model,
        train_dataset=train_dataset,
        val_dataset=train_dataset,
    )

    original_fit = flower_client.fit

    def audited_fit(parameters, config):
        weights, num_examples, metrics = original_fit(parameters, config)
        train_loss = float(metrics.get("loss", 1.0))
        quality_loss, quality_metric_value, holdout_metrics = compute_quality_holdout_loss(
            retriever,
            cid,
        )
        metrics["train_loss"] = train_loss
        metrics["loss"] = quality_loss
        metrics["quality_loss"] = quality_loss
        metrics["quality_metric_name"] = "log_mean_mrr_ndcg_holdout"
        metrics["quality_metric_value"] = quality_metric_value
        metrics["probe_size"] = len(CLIENT_QUALITY_HOLDOUTS[cid])
        metrics["probe_mrr"] = holdout_metrics.get("mrr", 0.0)
        metrics["probe_recall_at_k"] = holdout_metrics.get("recall_at_k", 0.0)
        metrics["probe_ndcg_at_k"] = holdout_metrics.get("ndcg_at_k", 0.0)
        metrics["train_loss_source"] = "lsr_training_loss"
        metrics["loss_source"] = "client_clean_holdout_retrieval"
        metrics["loss_stage"] = "post_local_training_clean_holdout"
        metrics["noise_mode"] = CURRENT_NOISE_MODE
        metrics["logical_cid"] = cid
        return weights, num_examples, metrics

    flower_client.fit = audited_fit

    logger.info(f"Client {cid}: Ready ({len(train_dataset)} examples)")
    return flower_client.to_client()


def weighted_average(metrics: list[Tuple[int, Metrics]]) -> Metrics:
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    avg_loss = sum(losses) / sum(examples)
    train_losses = [
        num_examples * float(m.get("train_loss", m["loss"]))
        for num_examples, m in metrics
    ]
    avg_train_loss = sum(train_losses) / sum(examples)

    ROUND_METRICS.append({
        "avg_loss": avg_loss,
        "avg_train_loss": avg_train_loss,
    })

    return {"loss": avg_loss}


def main(
    alpha: float,
    *,
    seed: int = SEED,
    noise_mode: str = NOISE_MODE,
    noise_ratio: float = NOISE_RATIO,
    num_rounds: int = NUM_ROUNDS,
    local_epochs: int = 1,
):
    global CLIENT_TRAIN_DATA, CLIENT_QUALITY_HOLDOUTS, KNOWLEDGE_STORE, EVAL_PAIRS
    global DOC_LOOKUP, ALPHA, REFERENCE_RETRIEVER
    global CURRENT_SEED, CURRENT_NUM_ROUNDS, CURRENT_LOCAL_EPOCHS
    global CURRENT_NOISE_MODE, NOISE_CONTEXT

    ALPHA = alpha
    CURRENT_SEED = seed
    CURRENT_NUM_ROUNDS = num_rounds
    CURRENT_LOCAL_EPOCHS = local_epochs
    CURRENT_NOISE_MODE = noise_mode

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TMP_TRAINING_DIR, exist_ok=True)

    run_slug = build_run_slug(
        alpha=alpha,
        seed=seed,
        noise_mode=noise_mode,
        noise_ratio=noise_ratio,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
    )
    log_file = os.path.join(LOG_DIR, f"log_{run_slug}.log")
    csv_path = os.path.join(CSV_DIR, f"results_{run_slug}.csv")
    manifest_path = os.path.join(CSV_DIR, f"manifest_{run_slug}.json")
    acceptance_path = os.path.join(CSV_DIR, f"acceptance_{run_slug}.json")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)

    print("=" * 70)
    print(
        f"Start | α={alpha:.1f} | seed={seed} | mode={noise_mode} | "
        f"ratio={noise_ratio:.2f} | rounds={num_rounds} | epochs={local_epochs}"
    )
    print(f"Device | runtime={get_runtime_device()}")
    print("=" * 70)

    data = setup_dataset(
        DATASET_NAME,
        max_train=MAX_TRAIN,
        max_eval=MAX_EVAL,
        max_docs=MAX_DOCS,
        seed=seed,
    )
    KNOWLEDGE_STORE = data["knowledge_store"]
    EVAL_PAIRS = data["eval_pairs"]
    DOC_LOOKUP = data["doc_lookup"]
    REFERENCE_RETRIEVER = data["retriever"]
    NOISE_CONTEXT = {}
    if noise_mode in {"cross_domain", "mixed"}:
        NOISE_CONTEXT["cross_domain_pool"] = build_cross_domain_response_pool(
            seed=seed
        )

    print(f"\nSplit train pairs across {NUM_CLIENTS} clients")
    CLIENT_TRAIN_DATA, CLIENT_QUALITY_HOLDOUTS = split_noisy(
        data["train_pairs"],
        NUM_CLIENTS,
        noise_ratio=noise_ratio,
        noisy_client=NOISY_CLIENT_ID,
        noise_mode=noise_mode,
    )
    split_summary = {
        "train": build_split_summary(CLIENT_TRAIN_DATA),
        "holdout": build_split_summary(CLIENT_QUALITY_HOLDOUTS),
    }

    retriever = REFERENCE_RETRIEVER
    pre_metrics = evaluate_retriever(retriever, KNOWLEDGE_STORE, EVAL_PAIRS, top_k=TOP_K)
    combined_holdout_pairs = flatten_holdout_pairs(CLIENT_QUALITY_HOLDOUTS)
    pre_probe_metrics = evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        combined_holdout_pairs,
        top_k=TOP_K,
    )
    print(
        "Pre-eval | "
        f"MRR={pre_metrics['mrr']:.4f} | "
        f"Recall@k={pre_metrics['recall_at_k']:.4f} | "
        f"NDCG@k={pre_metrics['ndcg_at_k']:.4f}"
    )
    print(
        "Holdout | "
        f"MRR={pre_probe_metrics['mrr']:.4f} | "
        f"Recall@k={pre_probe_metrics['recall_at_k']:.4f} | "
        f"NDCG@k={pre_probe_metrics['ndcg_at_k']:.4f}"
    )

    AcceleratorState._reset_state()
    model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
    initial_hash = hash_weights(model)
    ndarrays = _get_weights(model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    print(f"Initial model hash: {initial_hash[:16]}")

    strategy = QualityAwareFedAvg(
        alpha=alpha,
        focus_client_id=NOISY_CLIENT_ID,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=weighted_average,
        post_aggregation_evaluator=make_post_aggregation_evaluator(),
        initial_parameters=initial_parameters,
    )

    print("Running federated simulation...")

    ROUND_METRICS.clear()
    strategy.round_quality_info.clear()

    client_resources = get_client_resources()
    print(
        "Resources | "
        f"client_cpus={client_resources['num_cpus']} | "
        f"client_gpus={client_resources['num_gpus']}"
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    client_ids = sorted(CLIENT_TRAIN_DATA.keys(), key=client_sort_key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_csv_fieldnames(client_ids))
        writer.writeheader()

        for idx, qi in enumerate(strategy.round_quality_info):
            client_records = {
                record["cid"]: record for record in qi["client_records"]
            }
            row = {
                "round": qi["round"],
                "alpha": f"{alpha:.1f}",
                "seed": str(seed),
                "noise_mode": noise_mode,
                "noise_ratio": f"{noise_ratio:.2f}",
                "avg_loss": f"{qi['aggregated_loss']:.6f}",
                "avg_train_loss": "",
                "aggregated_delta_norm": f"{qi['aggregated_delta_norm']:.6f}",
                "aggregated_model_hash": qi["aggregated_model_hash"],
                "pre_mrr": f"{pre_metrics['mrr']:.6f}",
                "pre_recall_at_k": f"{pre_metrics['recall_at_k']:.6f}",
                "pre_ndcg_at_k": f"{pre_metrics['ndcg_at_k']:.6f}",
                "pre_probe_mrr": f"{pre_probe_metrics['mrr']:.6f}",
                "pre_probe_recall_at_k": f"{pre_probe_metrics['recall_at_k']:.6f}",
                "pre_probe_ndcg_at_k": f"{pre_probe_metrics['ndcg_at_k']:.6f}",
                "post_mrr": f"{qi['post_eval_metrics'].get('mrr', 0.0):.6f}",
                "post_recall_at_k": f"{qi['post_eval_metrics'].get('recall_at_k', 0.0):.6f}",
                "post_ndcg_at_k": f"{qi['post_eval_metrics'].get('ndcg_at_k', 0.0):.6f}",
            }

            if idx < len(ROUND_METRICS):
                row["avg_loss"] = f"{ROUND_METRICS[idx]['avg_loss']:.6f}"
                row["avg_train_loss"] = f"{ROUND_METRICS[idx]['avg_train_loss']:.6f}"

            for cid in client_ids:
                record = client_records.get(cid)
                if record is None:
                    continue
                row[f"client_{cid}_loss"] = f"{record['loss']:.6f}"
                row[f"client_{cid}_train_loss"] = f"{record['train_loss']:.6f}"
                row[f"client_{cid}_num_examples"] = str(record["num_examples"])
                row[f"client_{cid}_quality_score"] = f"{record['quality_score']:.6f}"
                row[f"client_{cid}_quality_metric_name"] = record["quality_metric_name"]
                row[f"client_{cid}_quality_metric_value"] = (
                    f"{record['quality_metric_value']:.6f}"
                )
                row[f"client_{cid}_size_weight"] = f"{record['size_weight']:.6f}"
                row[f"client_{cid}_weight"] = f"{record['combined_weight']:.6f}"
                row[f"client_{cid}_delta_norm"] = f"{record['delta_norm']:.6f}"
                row[f"client_{cid}_probe_size"] = str(record["probe_size"])
                row[f"client_{cid}_probe_mrr"] = f"{record['probe_mrr']:.6f}"
                row[f"client_{cid}_probe_recall_at_k"] = (
                    f"{record['probe_recall_at_k']:.6f}"
                )
                row[f"client_{cid}_probe_ndcg_at_k"] = (
                    f"{record['probe_ndcg_at_k']:.6f}"
                )
                row[f"client_{cid}_loss_source"] = record["loss_source"]
                row[f"client_{cid}_loss_stage"] = record["loss_stage"]
            writer.writerow(row)

    acceptance_report = evaluate_single_run_acceptance(
        strategy.round_quality_info,
        alpha=alpha,
    )
    with open(acceptance_path, "w", encoding="utf-8") as f:
        json.dump(acceptance_report, f, indent=2, sort_keys=True)

    write_run_manifest(
        manifest_path,
        alpha=alpha,
        seed=seed,
        noise_mode=noise_mode,
        noise_ratio=noise_ratio,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        initial_hash=initial_hash,
        pre_metrics=pre_metrics,
        pre_probe_metrics=pre_probe_metrics,
        split_summary=split_summary,
        csv_path=csv_path,
        log_path=log_file,
        acceptance_path=acceptance_path,
    )

    print(f"\n{'=' * 70}")
    print(f"✅ QA-FEDAVG NOISY EXPERIMENT COMPLETE (α={alpha})")
    print(f"📄 Results saved to: {csv_path}")
    print(f"🧾 Manifest saved to: {manifest_path}")
    print(f"✅ Acceptance report: {acceptance_path}")
    print(f"{'=' * 70}")

    logger.removeHandler(file_handler)
    file_handler.close()

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA-FedAvg Noisy Client Experiment")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Quality-aware alpha (0=FedAvg, 1=pure quality)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for dataset split and noise generation",
    )
    parser.add_argument(
        "--noise-mode",
        type=str,
        default=NOISE_MODE,
        choices=["shuffle", "hard_negative", "cross_domain", "random_negative", "mixed"],
        help="How to corrupt the noisy client's positives",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=NOISE_RATIO,
        help="Fraction of noisy client's data to corrupt",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help="Number of federated rounds",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs per client per round",
    )
    args = parser.parse_args()

    main(
        alpha=args.alpha,
        seed=args.seed,
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
    )
