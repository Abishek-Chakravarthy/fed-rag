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
from transformers.utils import logging as transformers_logging
from accelerate.state import AcceleratorState

from contrastive_trainer import ContrastiveFlowerClient

import flwr as fl
from flwr.common import Metrics
from flwr.common.parameter import ndarrays_to_parameters
from fed_rag.fl_tasks.huggingface import _get_weights

from prepare_beir_data import (
    MECHANISM_RETRIEVER_MODEL,
    RETRIEVER_MODEL,
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
NUM_CLIENTS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-6
DATASET_NAME = "nfcorpus"
BENCHMARK_MODE = "mechanism"
MAX_TRAIN = 4000 # Limits the training set to 4000 query-response pairs.
MAX_SHARED_QUALITY = 400 # Shared comparison pairs used for client weighting.
MAX_SERVER_VAL = 400 # Held-out validation pairs used for best-round selection.
MAX_FINAL_TEST = 500 # Final untouched test pairs used for the final report.
MAX_DOCS = 8000 # Limits the knowledge store to 8000 documents.
QUALITY_HOLDOUT_RATIO = 0.2 # Fraction of each client's clean data reserved for validation.
NOISE_RATIO = 0.8 # 80% of the noisy client's train split will be corrupted.
NOISE_MODE = "hard_negative" # The type of noise to introduce.
NOISY_CLIENT_ID = "4" # The client to introduce noise to.
CLIENT_SPLIT_MODE = "unequal" # Mechanism mode overrides this to equal client sizes.
NOISY_CLIENT_DATA_FRACTION = 0.4
QUALITY_BETA = 1.0
QUALITY_RANK_TOP_K = max(TOP_K * 3, 30)
MECHANISM_CLIENT_NOISE_MAP = {
    "0": 0.0,
    "1": 0.0,
    "2": 0.0,
    "3": 0.5,   # Step 3: was 0.3 — increased to widen quality gap for discrimination
    "4": 0.9,   # Step 3: was 0.7 — increased to widen quality gap for discrimination
}
MECHANISM_NOISE_MODE = "random_negative"  # Step 4a: was "shuffle" — shuffle is invisible to LSR (exp_03 proved identical models)
MECHANISM_CLIENT_SPLIT_MODE = "equal"
MECHANISM_RETRIEVER = MECHANISM_RETRIEVER_MODEL
ROBUSTNESS_RETRIEVER = RETRIEVER_MODEL
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
CLIENT_QUALITY_HOLDOUTS = {} # Clean per-client holdouts used for diagnostics.
SHARED_QUALITY_PAIRS = [] # Shared clean comparison set used for cross-client weighting.
KNOWLEDGE_STORE = None # The knowledge store for the RAG system.
SERVER_VAL_PAIRS = [] # Validation pairs used to pick the best global model.
FINAL_TEST_PAIRS = [] # Final held-out test pairs used for reporting.
DOC_LOOKUP = {} # A dictionary to look up documents in the knowledge store.
ROUND_METRICS = [] # A list to store the metrics for each round.
ALPHA = 0.5 # The alpha parameter for the QA-FedAvg algorithm.
CURRENT_SEED = SEED
CURRENT_NUM_ROUNDS = NUM_ROUNDS
CURRENT_LOCAL_EPOCHS = 1
CURRENT_BENCHMARK_MODE = BENCHMARK_MODE
CURRENT_NOISE_MODE = NOISE_MODE # A global tracker for how the "bad" client is corrupting its data (e.g., shuffle, cross_domain, etc.)
CURRENT_CLIENT_SPLIT_MODE = CLIENT_SPLIT_MODE
CURRENT_NOISY_CLIENT_DATA_FRACTION = NOISY_CLIENT_DATA_FRACTION
CURRENT_CLIENT_NOISE_MAP = dict(MECHANISM_CLIENT_NOISE_MAP)
CURRENT_RETRIEVER_MODEL = MECHANISM_RETRIEVER
NOISE_CONTEXT = {} # A global dictionary to hold extra data needed for certain noise modes (e.g., the pool of cross-domain documents).
REFERENCE_RETRIEVER = None # Frozen reference retriever used to construct hard in-domain negatives.


def get_runtime_device() -> str:
    import torch as _torch
    if _torch.cuda.is_available():
        return "cuda"
    if _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def mechanism_noise_map() -> dict[str, float]:
    return dict(MECHANISM_CLIENT_NOISE_MAP)


def mechanism_noise_summary() -> str:
    parts = []
    for cid, ratio in sorted(CURRENT_CLIENT_NOISE_MAP.items(), key=lambda item: client_sort_key(item[0])):
        parts.append(f"{cid}:{ratio:.1f}")
    return ", ".join(parts)


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
    if CURRENT_CLIENT_SPLIT_MODE == "equal":
        splits = split_iid(train_pairs, num_clients)
    elif CURRENT_CLIENT_SPLIT_MODE == "unequal":
        splits = split_unequal_noisy(
            train_pairs,
            num_clients,
            noisy_client=noisy_client,
            noisy_fraction=CURRENT_NOISY_CLIENT_DATA_FRACTION,
        )
    else:
        raise ValueError(f"Unsupported client split mode: {CURRENT_CLIENT_SPLIT_MODE}")

    train_splits, holdout_splits = reserve_clean_holdouts(splits)

    if CURRENT_BENCHMARK_MODE == "mechanism":
        rng = random.Random(CURRENT_SEED + 999)
        for cid in sorted(train_splits.keys(), key=client_sort_key):
            data = train_splits[cid].copy()
            client_noise_ratio = float(CURRENT_CLIENT_NOISE_MAP.get(cid, 0.0))
            num_corrupt = int(len(data) * client_noise_ratio)

            if num_corrupt > 0:
                original_responses = [p["response"] for p in data[:num_corrupt]]

                # Dispatch noise based on CURRENT_NOISE_MODE
                if CURRENT_NOISE_MODE == "shuffle":
                    replacement_responses = deranged_shuffle(original_responses, rng)
                elif CURRENT_NOISE_MODE == "random_negative":
                    replacement_responses = sample_random_negative_responses(
                        count=num_corrupt,
                        rng=rng,
                        doc_lookup=DOC_LOOKUP,
                        original_responses=original_responses,
                    )
                elif CURRENT_NOISE_MODE == "hard_negative":
                    replacement_responses = sample_hard_negative_responses(
                        pairs=data[:num_corrupt],
                        rng=rng,
                    )
                elif CURRENT_NOISE_MODE == "cross_domain":
                    pool = NOISE_CONTEXT.get("cross_domain_pool", [])
                    if not pool:
                        raise ValueError("Cross-domain noise requested but no pool is available.")
                    replacement_responses = [rng.choice(pool) for _ in range(num_corrupt)]
                else:
                    # Default fallback to shuffle
                    replacement_responses = deranged_shuffle(original_responses, rng)

                corrupted_data = []
                for i in range(num_corrupt):
                    pair = dict(data[i])
                    pair["response"] = replacement_responses[i]
                    corrupted_data.append(pair)
                corrupted_data.extend(data[num_corrupt:])
                train_splits[cid] = corrupted_data
                status = f"{num_corrupt} CORRUPTED ({client_noise_ratio*100:.0f}% {CURRENT_NOISE_MODE} noise)"
            else:
                status = "CLEAN"

            print(
                f"  Client {cid}: {len(train_splits[cid])} train + "
                f"{len(holdout_splits[cid])} holdout ({status})"
            )

        return train_splits, holdout_splits

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
        "benchmark_mode",
        "retriever_model",
        "noisy_client_id",
        "client_split_mode",
        "noise_mode",
        "noise_ratio",
        "avg_loss",
        "avg_train_loss",
        "aggregated_delta_norm",
        "aggregated_model_hash",
        "pre_server_val_mrr",
        "pre_server_val_recall_at_k",
        "pre_server_val_ndcg_at_k",
        "pre_final_test_mrr",
        "pre_final_test_recall_at_k",
        "pre_final_test_ndcg_at_k",
        "pre_probe_mrr",
        "pre_probe_recall_at_k",
        "pre_probe_ndcg_at_k",
        "pre_shared_quality_mrr",
        "pre_shared_quality_recall_at_k",
        "pre_shared_quality_ndcg_at_k",
        "pre_shared_quality_mean_rank",
        "server_val_mrr",
        "server_val_recall_at_k",
        "server_val_ndcg_at_k",
        "post_mrr",
        "post_recall_at_k",
        "post_ndcg_at_k",
        "best_round_so_far",
        "best_server_val_mrr_so_far",
        "best_server_val_ndcg_so_far",
        "selected_best_round",
        "final_test_mrr",
        "final_test_recall_at_k",
        "final_test_ndcg_at_k",
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
                f"client_{cid}_configured_noise_ratio",
                f"client_{cid}_delta_norm",
                f"client_{cid}_probe_size",
                f"client_{cid}_probe_mrr",
                f"client_{cid}_probe_recall_at_k",
                f"client_{cid}_probe_ndcg_at_k",
                f"client_{cid}_shared_quality_size",
                f"client_{cid}_shared_quality_mrr",
                f"client_{cid}_shared_quality_recall_at_k",
                f"client_{cid}_shared_quality_ndcg_at_k",
                f"client_{cid}_shared_quality_mean_rank_before",
                f"client_{cid}_shared_quality_mean_rank_after",
                f"client_{cid}_shared_quality_mean_rank_delta",
                f"client_{cid}_shared_quality_mean_positive_delta",
                f"client_{cid}_shared_quality_mean_negative_delta",
                f"client_{cid}_shared_quality_degradation_rate",
                f"client_{cid}_shared_quality_improvement_rate",
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
    benchmark_mode: str,
    seed: int,
    noise_mode: str,
    noise_ratio: float,
    num_rounds: int,
    local_epochs: int,
) -> str:
    benchmark_slug = benchmark_mode.replace("-", "_")
    mode = noise_mode.replace("-", "_")
    ratio = f"{noise_ratio:.2f}".replace(".", "p")
    return (
        f"alpha_{alpha:.1f}_track_{benchmark_slug}_seed_{seed}_mode_{mode}_"
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
        # Rebuild the same retriever architecture used by clients so the
        # aggregated query-encoder weights map onto an identical state_dict.
        retriever = create_retriever(CURRENT_RETRIEVER_MODEL)
        set_retriever_weights(retriever, aggregated_ndarrays)
        return evaluate_retriever(
            retriever,
            KNOWLEDGE_STORE,
            SERVER_VAL_PAIRS,
            top_k=TOP_K,
        )

    return evaluator


def flatten_holdout_pairs(holdout_splits):
    pairs = []
    for cid in sorted(holdout_splits.keys(), key=client_sort_key):
        pairs.extend(holdout_splits[cid])
    return pairs


def compute_doc_ranks(retriever, pairs, *, top_k: int = QUALITY_RANK_TOP_K):
    ranks = []
    for pair in pairs:
        query_emb = retriever.encode_query(pair["query"])[0].tolist()
        results = KNOWLEDGE_STORE.retrieve(query_emb, top_k=top_k)

        rank = top_k + 1
        for idx, (_score, node) in enumerate(results, 1):
            if str(node.metadata.get("doc_id", "")) == str(pair["doc_id"]):
                rank = idx
                break
        ranks.append(rank)
    return ranks


def summarize_rank_displacement(before_ranks, after_ranks):
    if not before_ranks or not after_ranks:
        return {
            "quality_loss": 0.0,
            "quality_metric_value": 0.0,
            "mean_rank_before": 0.0,
            "mean_rank_after": 0.0,
            "mean_rank_delta": 0.0,
            "mean_positive_delta": 0.0,
            "mean_negative_delta": 0.0,
            "degradation_rate": 0.0,
            "improvement_rate": 0.0,
        }

    before_arr = np.asarray(before_ranks, dtype=float)
    after_arr = np.asarray(after_ranks, dtype=float)
    deltas = after_arr - before_arr
    positive = np.maximum(deltas, 0.0)
    negative = np.maximum(-deltas, 0.0)
    degradation_rate = float(np.mean(deltas > 0))
    improvement_rate = float(np.mean(deltas < 0))
    mean_positive_delta = float(np.mean(positive))
    mean_negative_delta = float(np.mean(negative))

    # Lower is better: clients that push correct documents down in rank should
    # incur a larger quality loss, while clients that improve ranking get credit.
    quality_loss = float(mean_positive_delta - 0.5 * mean_negative_delta)
    quality_metric_value = float(-quality_loss)

    return {
        "quality_loss": quality_loss,
        "quality_metric_value": quality_metric_value,
        "mean_rank_before": float(np.mean(before_arr)),
        "mean_rank_after": float(np.mean(after_arr)),
        "mean_rank_delta": float(np.mean(deltas)),
        "mean_positive_delta": mean_positive_delta,
        "mean_negative_delta": mean_negative_delta,
        "degradation_rate": degradation_rate,
        "improvement_rate": improvement_rate,
    }


def compute_shared_quality_signal(retriever, baseline_ranks):
    after_ranks = compute_doc_ranks(
        retriever,
        SHARED_QUALITY_PAIRS,
        top_k=QUALITY_RANK_TOP_K,
    )
    summary = summarize_rank_displacement(baseline_ranks, after_ranks)
    comparison_metrics = evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        SHARED_QUALITY_PAIRS,
        top_k=TOP_K,
    )
    summary.update(
        {
            "shared_mrr": comparison_metrics.get("mrr", 0.0),
            "shared_recall_at_k": comparison_metrics.get("recall_at_k", 0.0),
            "shared_ndcg_at_k": comparison_metrics.get("ndcg_at_k", 0.0),
            "shared_size": len(SHARED_QUALITY_PAIRS),
        }
    )
    return summary


def compute_client_holdout_metrics(retriever, cid: str):
    holdout_pairs = CLIENT_QUALITY_HOLDOUTS[cid]
    return evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        holdout_pairs,
        top_k=TOP_K,
    )


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
            # Use the highest-ranked wrong document to make the corruption
            # consistently challenging instead of randomly mild.
            responses.append(candidates[0])
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
    benchmark_mode: str,
    seed: int,
    retriever_model: str,
    noise_mode: str,
    noise_ratio: float,
    num_rounds: int,
    local_epochs: int,
    initial_hash: str,
    pre_metrics: dict,
    pre_shared_quality_metrics: dict,
    pre_holdout_metrics: dict,
    pre_final_test_metrics: dict,
    final_test_metrics: dict,
    best_round: int,
    split_summary: dict,
    csv_path: str,
    log_path: str,
    acceptance_path: str,
    beta: float,
):
    manifest = {
        "experiment_name": "qa_fedavg_noisy_client",
        "canonical_results_csv": csv_path,
        "canonical_log_file": log_path,
        "acceptance_report": acceptance_path,
        "alpha": alpha,
        "benchmark_mode": benchmark_mode,
        "seed": seed,
        "dataset_name": DATASET_NAME,
        "retriever_model": retriever_model,
        "num_rounds": num_rounds,
        "num_clients": NUM_CLIENTS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "local_epochs": local_epochs,
        "max_train_pairs": MAX_TRAIN,
        "max_shared_quality_pairs": MAX_SHARED_QUALITY,
        "max_server_val_pairs": MAX_SERVER_VAL,
        "max_final_test_pairs": MAX_FINAL_TEST,
        "max_corpus_docs": MAX_DOCS,
        "shared_quality_pairs_count": len(SHARED_QUALITY_PAIRS),
        "quality_holdout_ratio": QUALITY_HOLDOUT_RATIO,
        "client_split_mode": CURRENT_CLIENT_SPLIT_MODE,
        "noisy_client_data_fraction": CURRENT_NOISY_CLIENT_DATA_FRACTION,
        "quality_beta": beta,
        "quality_rank_top_k": QUALITY_RANK_TOP_K,
        "top_k": TOP_K,
        "noise_ratio": noise_ratio,
        "noise_mode": noise_mode,
        "noisy_client_id": NOISY_CLIENT_ID,
        "client_noise_map": dict(CURRENT_CLIENT_NOISE_MAP),
        "initial_model_hash": initial_hash,
        "pre_metrics": pre_metrics,
        "pre_shared_quality_metrics": pre_shared_quality_metrics,
        "pre_holdout_metrics": pre_holdout_metrics,
        "pre_final_test_metrics": pre_final_test_metrics,
        "final_test_metrics": final_test_metrics,
        "best_round": best_round,
        "client_split_summary": split_summary,
        "train_split_hash": hash_jsonable(split_summary),
        "csv_schema_version": 5,
        "quality_signal": {
            "strategy_metric_key": "loss",
            "client_metric_source": "delta_mrr",
            "local_training_objective": "MultipleNegativesRankingLoss (InfoNCE / contrastive)",
            "trainer_return_value": "MRR_before - MRR_after (on 400-pair shared probe set)",
            "loss_stage": "post_local_training_shared_comparison",
            "objective": "MRR degradation on shared probe — positive = degraded, negative = improved",
        },
        "model_selection": {
            "criterion": "best_server_val_mrr_then_ndcg",
            "selected_best_round": best_round,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def evaluate_single_run_acceptance(round_infos, alpha: float, *, benchmark_mode: str):
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
        "lower_quality_clients_downweighted": {
            "applicable": alpha > 0.0 and benchmark_mode == "mechanism",
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
        quality_order_rounds = 0
        noisy_clients = [
            cid for cid, ratio in CURRENT_CLIENT_NOISE_MAP.items() if ratio > 0.0
        ]
        clean_clients = [
            cid for cid, ratio in CURRENT_CLIENT_NOISE_MAP.items() if ratio == 0.0
        ]
        for round_info in round_infos:
            noisiest = max(
                round_info["client_records"],
                key=lambda record: record["loss"],
            )
            if noisiest["combined_weight"] < noisiest["size_weight"]:
                downweighted_rounds += 1
            if benchmark_mode == "mechanism":
                record_map = {
                    record["cid"]: record for record in round_info["client_records"]
                }
                noisy_weights = [
                    record_map[cid]["combined_weight"]
                    for cid in noisy_clients
                    if cid in record_map
                ]
                clean_weights = [
                    record_map[cid]["combined_weight"]
                    for cid in clean_clients
                    if cid in record_map
                ]
                if noisy_weights and clean_weights and max(noisy_weights) < min(clean_weights):
                    quality_order_rounds += 1
        report["higher_loss_clients_downweighted"]["passed"] = (
            downweighted_rounds == len(round_infos)
        )
        report["higher_loss_clients_downweighted"]["evidence"] = {
            "downweighted_rounds": downweighted_rounds,
            "total_rounds": len(round_infos),
        }
        if benchmark_mode == "mechanism":
            report["lower_quality_clients_downweighted"]["passed"] = (
                quality_order_rounds == len(round_infos)
            )
            report["lower_quality_clients_downweighted"]["evidence"] = {
                "quality_order_rounds": quality_order_rounds,
                "total_rounds": len(round_infos),
                "client_noise_map": dict(CURRENT_CLIENT_NOISE_MAP),
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

    logger.info(f"Client {cid}: Creating retriever...")

    retriever = create_retriever(CURRENT_RETRIEVER_MODEL)

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

    # model must be the SAME object as retriever.query_encoder so that
    # audited_fit's quality signal evaluates post-training retriever weights.
    model = retriever.query_encoder if retriever.query_encoder else retriever.encoder

    contrastive_client = ContrastiveFlowerClient(
        model=model,
        train_dataset=train_dataset,
        training_args=training_args,
    )

    original_fit = contrastive_client.fit

    def audited_fit(parameters, config):
        set_retriever_weights(retriever, parameters)
        # Compute MRR on shared probe BEFORE local training.
        mrr_before_metrics = evaluate_retriever(
            retriever, KNOWLEDGE_STORE, SHARED_QUALITY_PAIRS, top_k=TOP_K,
        )
        mrr_before = mrr_before_metrics.get("mrr", 0.0)
        # Also capture rank state for the supplementary displacement columns.
        baseline_shared_ranks = compute_doc_ranks(
            retriever, SHARED_QUALITY_PAIRS, top_k=QUALITY_RANK_TOP_K,
        )
        weights, num_examples, metrics = original_fit(parameters, config)
        train_loss = float(metrics.get("loss", 1.0))
        # Compute MRR on shared probe AFTER local training.
        mrr_after_metrics = evaluate_retriever(
            retriever, KNOWLEDGE_STORE, SHARED_QUALITY_PAIRS, top_k=TOP_K,
        )
        mrr_after = mrr_after_metrics.get("mrr", 0.0)
        # delta_mrr > 0 means retriever degraded; < 0 means it improved.
        # Used directly as quality loss: server downweights clients with positive delta_mrr.
        delta_mrr = mrr_before - mrr_after
        quality_summary = compute_shared_quality_signal(retriever, baseline_shared_ranks)
        holdout_metrics = compute_client_holdout_metrics(retriever, cid)
        metrics["train_loss"] = train_loss
        metrics["loss"] = delta_mrr
        metrics["quality_loss"] = delta_mrr
        metrics["quality_metric_name"] = "delta_mrr"
        metrics["quality_metric_value"] = -delta_mrr  # positive = improvement
        metrics["probe_size"] = len(CLIENT_QUALITY_HOLDOUTS[cid])
        metrics["probe_mrr"] = holdout_metrics.get("mrr", 0.0)
        metrics["probe_recall_at_k"] = holdout_metrics.get("recall_at_k", 0.0)
        metrics["probe_ndcg_at_k"] = holdout_metrics.get("ndcg_at_k", 0.0)
        metrics["shared_quality_size"] = len(SHARED_QUALITY_PAIRS)
        metrics["shared_quality_mrr"] = mrr_after
        metrics["shared_quality_recall_at_k"] = mrr_after_metrics.get("recall_at_k", 0.0)
        metrics["shared_quality_ndcg_at_k"] = mrr_after_metrics.get("ndcg_at_k", 0.0)
        metrics["shared_quality_mean_rank_before"] = quality_summary["mean_rank_before"]
        metrics["shared_quality_mean_rank_after"] = quality_summary["mean_rank_after"]
        metrics["shared_quality_mean_rank_delta"] = quality_summary["mean_rank_delta"]
        metrics["shared_quality_mean_positive_delta"] = quality_summary["mean_positive_delta"]
        metrics["shared_quality_mean_negative_delta"] = quality_summary["mean_negative_delta"]
        metrics["shared_quality_degradation_rate"] = quality_summary["degradation_rate"]
        metrics["shared_quality_improvement_rate"] = quality_summary["improvement_rate"]
        metrics["train_loss_source"] = "contrastive_mnr_loss"
        metrics["loss_source"] = "delta_mrr"
        metrics["loss_stage"] = "post_local_training_shared_comparison"
        metrics["noise_mode"] = CURRENT_NOISE_MODE
        metrics["logical_cid"] = cid
        return weights, num_examples, metrics

    contrastive_client.fit = audited_fit

    logger.info(f"Client {cid}: Ready ({len(train_dataset)} examples)")
    return contrastive_client.to_client()


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


def resolve_benchmark_config(
    *,
    benchmark_mode: str,
    noise_mode: str,
    noise_ratio: float,
    client_split_mode: str,
    noisy_client_fraction: float,
) -> dict:
    if benchmark_mode == "mechanism":
        return {
            "benchmark_mode": benchmark_mode,
            "noise_mode": MECHANISM_NOISE_MODE,
            "noise_ratio": 0.0,
            "client_split_mode": MECHANISM_CLIENT_SPLIT_MODE,
            "noisy_client_fraction": 0.0,
            "client_noise_map": mechanism_noise_map(),
            "retriever_model": MECHANISM_RETRIEVER,
            "focus_client_id": "4",
        }

    return {
        "benchmark_mode": benchmark_mode,
        "noise_mode": noise_mode,
        "noise_ratio": noise_ratio,
        "client_split_mode": client_split_mode,
        "noisy_client_fraction": noisy_client_fraction,
        "client_noise_map": {NOISY_CLIENT_ID: noise_ratio},
        "retriever_model": ROBUSTNESS_RETRIEVER,
        "focus_client_id": NOISY_CLIENT_ID,
    }


def main(
    alpha: float,
    *,
    benchmark_mode: str = BENCHMARK_MODE,
    seed: int = SEED,
    noise_mode: str = NOISE_MODE,
    noise_ratio: float = NOISE_RATIO,
    num_rounds: int = NUM_ROUNDS,
    local_epochs: int = 1,
    client_split_mode: str = CLIENT_SPLIT_MODE,
    noisy_client_fraction: float = NOISY_CLIENT_DATA_FRACTION,
    beta: float = QUALITY_BETA,
):
    global CLIENT_TRAIN_DATA, CLIENT_QUALITY_HOLDOUTS, SHARED_QUALITY_PAIRS
    global KNOWLEDGE_STORE, SERVER_VAL_PAIRS, FINAL_TEST_PAIRS
    global DOC_LOOKUP, ALPHA, REFERENCE_RETRIEVER
    global CURRENT_SEED, CURRENT_NUM_ROUNDS, CURRENT_LOCAL_EPOCHS
    global CURRENT_BENCHMARK_MODE, CURRENT_NOISE_MODE, CURRENT_CLIENT_SPLIT_MODE
    global CURRENT_NOISY_CLIENT_DATA_FRACTION, CURRENT_CLIENT_NOISE_MAP
    global CURRENT_RETRIEVER_MODEL, NOISE_CONTEXT

    ALPHA = alpha
    CURRENT_SEED = seed
    CURRENT_NUM_ROUNDS = num_rounds
    CURRENT_LOCAL_EPOCHS = local_epochs
    config = resolve_benchmark_config(
        benchmark_mode=benchmark_mode,
        noise_mode=noise_mode,
        noise_ratio=noise_ratio,
        client_split_mode=client_split_mode,
        noisy_client_fraction=noisy_client_fraction,
    )
    CURRENT_BENCHMARK_MODE = config["benchmark_mode"]
    CURRENT_NOISE_MODE = config["noise_mode"]
    CURRENT_CLIENT_SPLIT_MODE = config["client_split_mode"]
    CURRENT_NOISY_CLIENT_DATA_FRACTION = config["noisy_client_fraction"]
    CURRENT_CLIENT_NOISE_MAP = dict(config["client_noise_map"])
    CURRENT_RETRIEVER_MODEL = str(config["retriever_model"])
    effective_noise_ratio = float(config["noise_ratio"])
    focus_client_id = str(config["focus_client_id"])

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TMP_TRAINING_DIR, exist_ok=True)

    run_slug = build_run_slug(
        alpha=alpha,
        benchmark_mode=CURRENT_BENCHMARK_MODE,
        seed=seed,
        noise_mode=CURRENT_NOISE_MODE,
        noise_ratio=effective_noise_ratio,
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
        f"Start | α={alpha:.1f} | seed={seed} | track={CURRENT_BENCHMARK_MODE} | "
        f"mode={CURRENT_NOISE_MODE} | ratio={effective_noise_ratio:.2f} | "
        f"rounds={num_rounds} | epochs={local_epochs}"
    )
    print(
        f"Config | clients={NUM_CLIENTS} | split={CURRENT_CLIENT_SPLIT_MODE} | "
        f"noisy_fraction={CURRENT_NOISY_CLIENT_DATA_FRACTION:.2f} | "
        f"quality_beta={beta:.1f}"
    )
    print(f"Retriever | model={CURRENT_RETRIEVER_MODEL}")
    if CURRENT_BENCHMARK_MODE == "mechanism":
        print(f"NoiseMap | {mechanism_noise_summary()}")
    print(f"Device | runtime={get_runtime_device()}")
    print("=" * 70)

    data = setup_dataset(
        DATASET_NAME,
        max_train=MAX_TRAIN,
        max_shared_quality=MAX_SHARED_QUALITY,
        max_server_val=MAX_SERVER_VAL,
        max_final_test=MAX_FINAL_TEST,
        max_docs=MAX_DOCS,
        seed=seed,
        retriever_model=CURRENT_RETRIEVER_MODEL,
    )
    KNOWLEDGE_STORE = data["knowledge_store"]
    SERVER_VAL_PAIRS = data["server_val_pairs"]
    FINAL_TEST_PAIRS = data["final_test_pairs"]
    SHARED_QUALITY_PAIRS = data["shared_quality_pairs"]
    DOC_LOOKUP = data["doc_lookup"]
    REFERENCE_RETRIEVER = data["retriever"]
    NOISE_CONTEXT = {}
    if CURRENT_NOISE_MODE in {"cross_domain", "mixed"}:
        NOISE_CONTEXT["cross_domain_pool"] = build_cross_domain_response_pool(
            seed=seed
        )

    print(f"\nSplit train pairs across {NUM_CLIENTS} clients")
    CLIENT_TRAIN_DATA, CLIENT_QUALITY_HOLDOUTS = split_noisy(
        data["train_pairs"],
        NUM_CLIENTS,
        noise_ratio=effective_noise_ratio,
        noisy_client=NOISY_CLIENT_ID,
        noise_mode=CURRENT_NOISE_MODE,
    )
    split_summary = {
        "train": build_split_summary(CLIENT_TRAIN_DATA),
        "holdout": build_split_summary(CLIENT_QUALITY_HOLDOUTS),
    }

    retriever = REFERENCE_RETRIEVER
    pre_server_val_metrics = evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        SERVER_VAL_PAIRS,
        top_k=TOP_K,
    )
    pre_final_test_metrics = evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        FINAL_TEST_PAIRS,
        top_k=TOP_K,
    )
    pre_probe_metrics = evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        flatten_holdout_pairs(CLIENT_QUALITY_HOLDOUTS),
        top_k=TOP_K,
    )
    pre_shared_quality_metrics = evaluate_retriever(
        retriever,
        KNOWLEDGE_STORE,
        SHARED_QUALITY_PAIRS,
        top_k=TOP_K,
    )
    pre_shared_ranks = compute_doc_ranks(
        retriever,
        SHARED_QUALITY_PAIRS,
        top_k=QUALITY_RANK_TOP_K,
    )
    print(
        "ServerV | "
        f"MRR={pre_server_val_metrics['mrr']:.4f} | "
        f"Recall@k={pre_server_val_metrics['recall_at_k']:.4f} | "
        f"NDCG@k={pre_server_val_metrics['ndcg_at_k']:.4f}"
    )
    print(
        "Test   | "
        f"MRR={pre_final_test_metrics['mrr']:.4f} | "
        f"Recall@k={pre_final_test_metrics['recall_at_k']:.4f} | "
        f"NDCG@k={pre_final_test_metrics['ndcg_at_k']:.4f}"
    )
    print(
        "Holdout | "
        f"MRR={pre_probe_metrics['mrr']:.4f} | "
        f"Recall@k={pre_probe_metrics['recall_at_k']:.4f} | "
        f"NDCG@k={pre_probe_metrics['ndcg_at_k']:.4f}"
    )
    print(
        "Shared  | "
        f"MRR={pre_shared_quality_metrics['mrr']:.4f} | "
        f"Recall@k={pre_shared_quality_metrics['recall_at_k']:.4f} | "
        f"NDCG@k={pre_shared_quality_metrics['ndcg_at_k']:.4f} | "
        f"MeanRank={float(np.mean(pre_shared_ranks)):.2f}"
    )

    AcceleratorState._reset_state()
    model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
    initial_hash = hash_weights(model)
    ndarrays = _get_weights(model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    print(f"Initial model hash: {initial_hash[:16]}")

    strategy = QualityAwareFedAvg(
        alpha=alpha,
        quality_beta=beta,
        focus_client_id=focus_client_id,
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

    selected_ndarrays = strategy.best_global_ndarrays or strategy.last_global_ndarrays
    final_test_metrics = {"mrr": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}
    selected_best_round = strategy.best_round or (
        strategy.round_quality_info[-1]["round"] if strategy.round_quality_info else 0
    )
    if selected_ndarrays is not None:
        final_retriever = create_retriever(CURRENT_RETRIEVER_MODEL)
        set_retriever_weights(final_retriever, selected_ndarrays)
        final_test_metrics = evaluate_retriever(
            final_retriever,
            KNOWLEDGE_STORE,
            FINAL_TEST_PAIRS,
            top_k=TOP_K,
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
                "benchmark_mode": CURRENT_BENCHMARK_MODE,
                "retriever_model": CURRENT_RETRIEVER_MODEL,
                "noisy_client_id": NOISY_CLIENT_ID,
                "client_split_mode": CURRENT_CLIENT_SPLIT_MODE,
                "noise_mode": CURRENT_NOISE_MODE,
                "noise_ratio": f"{effective_noise_ratio:.2f}",
                "avg_loss": f"{qi['aggregated_loss']:.6f}",
                "avg_train_loss": "",
                "aggregated_delta_norm": f"{qi['aggregated_delta_norm']:.6f}",
                "aggregated_model_hash": qi["aggregated_model_hash"],
                "pre_server_val_mrr": f"{pre_server_val_metrics['mrr']:.6f}",
                "pre_server_val_recall_at_k": f"{pre_server_val_metrics['recall_at_k']:.6f}",
                "pre_server_val_ndcg_at_k": f"{pre_server_val_metrics['ndcg_at_k']:.6f}",
                "pre_final_test_mrr": f"{pre_final_test_metrics['mrr']:.6f}",
                "pre_final_test_recall_at_k": f"{pre_final_test_metrics['recall_at_k']:.6f}",
                "pre_final_test_ndcg_at_k": f"{pre_final_test_metrics['ndcg_at_k']:.6f}",
                "pre_probe_mrr": f"{pre_probe_metrics['mrr']:.6f}",
                "pre_probe_recall_at_k": f"{pre_probe_metrics['recall_at_k']:.6f}",
                "pre_probe_ndcg_at_k": f"{pre_probe_metrics['ndcg_at_k']:.6f}",
                "pre_shared_quality_mrr": f"{pre_shared_quality_metrics['mrr']:.6f}",
                "pre_shared_quality_recall_at_k": f"{pre_shared_quality_metrics['recall_at_k']:.6f}",
                "pre_shared_quality_ndcg_at_k": f"{pre_shared_quality_metrics['ndcg_at_k']:.6f}",
                "pre_shared_quality_mean_rank": f"{float(np.mean(pre_shared_ranks)):.6f}",
                "server_val_mrr": f"{qi['post_eval_metrics'].get('mrr', 0.0):.6f}",
                "server_val_recall_at_k": f"{qi['post_eval_metrics'].get('recall_at_k', 0.0):.6f}",
                "server_val_ndcg_at_k": f"{qi['post_eval_metrics'].get('ndcg_at_k', 0.0):.6f}",
                "post_mrr": f"{qi['post_eval_metrics'].get('mrr', 0.0):.6f}",
                "post_recall_at_k": f"{qi['post_eval_metrics'].get('recall_at_k', 0.0):.6f}",
                "post_ndcg_at_k": f"{qi['post_eval_metrics'].get('ndcg_at_k', 0.0):.6f}",
                "best_round_so_far": str(qi.get("best_round_so_far", "")),
                "best_server_val_mrr_so_far": (
                    f"{qi.get('best_post_eval_metrics_so_far', {}).get('mrr', 0.0):.6f}"
                ),
                "best_server_val_ndcg_so_far": (
                    f"{qi.get('best_post_eval_metrics_so_far', {}).get('ndcg_at_k', 0.0):.6f}"
                ),
                "selected_best_round": str(selected_best_round),
                "final_test_mrr": f"{final_test_metrics.get('mrr', 0.0):.6f}",
                "final_test_recall_at_k": f"{final_test_metrics.get('recall_at_k', 0.0):.6f}",
                "final_test_ndcg_at_k": f"{final_test_metrics.get('ndcg_at_k', 0.0):.6f}",
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
                row[f"client_{cid}_configured_noise_ratio"] = (
                    f"{CURRENT_CLIENT_NOISE_MAP.get(cid, 0.0):.2f}"
                )
                row[f"client_{cid}_delta_norm"] = f"{record['delta_norm']:.6f}"
                row[f"client_{cid}_probe_size"] = str(record["probe_size"])
                row[f"client_{cid}_probe_mrr"] = f"{record['probe_mrr']:.6f}"
                row[f"client_{cid}_probe_recall_at_k"] = (
                    f"{record['probe_recall_at_k']:.6f}"
                )
                row[f"client_{cid}_probe_ndcg_at_k"] = (
                    f"{record['probe_ndcg_at_k']:.6f}"
                )
                row[f"client_{cid}_shared_quality_size"] = str(record["shared_quality_size"])
                row[f"client_{cid}_shared_quality_mrr"] = (
                    f"{record['shared_quality_mrr']:.6f}"
                )
                row[f"client_{cid}_shared_quality_recall_at_k"] = (
                    f"{record['shared_quality_recall_at_k']:.6f}"
                )
                row[f"client_{cid}_shared_quality_ndcg_at_k"] = (
                    f"{record['shared_quality_ndcg_at_k']:.6f}"
                )
                row[f"client_{cid}_shared_quality_mean_rank_before"] = (
                    f"{record['shared_quality_mean_rank_before']:.6f}"
                )
                row[f"client_{cid}_shared_quality_mean_rank_after"] = (
                    f"{record['shared_quality_mean_rank_after']:.6f}"
                )
                row[f"client_{cid}_shared_quality_mean_rank_delta"] = (
                    f"{record['shared_quality_mean_rank_delta']:.6f}"
                )
                row[f"client_{cid}_shared_quality_mean_positive_delta"] = (
                    f"{record['shared_quality_mean_positive_delta']:.6f}"
                )
                row[f"client_{cid}_shared_quality_mean_negative_delta"] = (
                    f"{record['shared_quality_mean_negative_delta']:.6f}"
                )
                row[f"client_{cid}_shared_quality_degradation_rate"] = (
                    f"{record['shared_quality_degradation_rate']:.6f}"
                )
                row[f"client_{cid}_shared_quality_improvement_rate"] = (
                    f"{record['shared_quality_improvement_rate']:.6f}"
                )
                row[f"client_{cid}_loss_source"] = record["loss_source"]
                row[f"client_{cid}_loss_stage"] = record["loss_stage"]
            writer.writerow(row)

    acceptance_report = evaluate_single_run_acceptance(
        strategy.round_quality_info,
        alpha=alpha,
        benchmark_mode=CURRENT_BENCHMARK_MODE,
    )
    with open(acceptance_path, "w", encoding="utf-8") as f:
        json.dump(acceptance_report, f, indent=2, sort_keys=True)

    write_run_manifest(
        manifest_path,
        alpha=alpha,
        benchmark_mode=CURRENT_BENCHMARK_MODE,
        seed=seed,
        retriever_model=CURRENT_RETRIEVER_MODEL,
        noise_mode=CURRENT_NOISE_MODE,
        noise_ratio=effective_noise_ratio,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        initial_hash=initial_hash,
        pre_metrics=pre_server_val_metrics,
        pre_shared_quality_metrics=pre_shared_quality_metrics,
        pre_holdout_metrics=pre_probe_metrics,
        pre_final_test_metrics=pre_final_test_metrics,
        final_test_metrics=final_test_metrics,
        best_round=selected_best_round,
        split_summary=split_summary,
        csv_path=csv_path,
        log_path=log_file,
        acceptance_path=acceptance_path,
        beta=beta,
    )

    print(f"\n{'=' * 70}")
    print(
        f"QA-FEDAVG EXPERIMENT COMPLETE | α={alpha:.1f} | "
        f"track={CURRENT_BENCHMARK_MODE}"
    )
    print(
        f"🏁 Best round={selected_best_round} | "
        f"final_test_mrr={final_test_metrics.get('mrr', 0.0):.4f} | "
        f"final_test_ndcg={final_test_metrics.get('ndcg_at_k', 0.0):.4f}"
    )
    print(f"📄 Results saved to: {csv_path}")
    print(f"🧾 Manifest saved to: {manifest_path}")
    print(f"✅ Acceptance report: {acceptance_path}")
    print(f"{'=' * 70}")

    logger.removeHandler(file_handler)
    file_handler.close()

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA-FedAvg mechanism/stress experiment")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Quality-aware alpha (0=FedAvg, 1=pure quality)",
    )
    parser.add_argument(
        "--benchmark-mode",
        type=str,
        default=BENCHMARK_MODE,
        choices=["mechanism", "stress"],
        help="Use 'mechanism' for the controlled quality-ladder benchmark and 'stress' for the harder single-noisy-client benchmark",
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
    parser.add_argument(
        "--beta",
        type=float,
        default=QUALITY_BETA,
        help="Temperature parameter for quality score softmax",
    )
    parser.add_argument(
        "--client-split-mode",
        type=str,
        default=CLIENT_SPLIT_MODE,
        choices=["equal", "unequal"],
        help="Client data split mode. Mechanism mode overrides this to equal splits.",
    )
    parser.add_argument(
        "--noisy-client-fraction",
        type=float,
        default=NOISY_CLIENT_DATA_FRACTION,
        help="Fraction of training pairs assigned to the noisy client when using unequal splits",
    )
    args = parser.parse_args()

    main(
        alpha=args.alpha,
        benchmark_mode=args.benchmark_mode,
        seed=args.seed,
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_split_mode=args.client_split_mode,
        noisy_client_fraction=args.noisy_client_fraction,
        beta=args.beta,
    )
