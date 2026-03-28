import argparse
import csv
import hashlib
import json
import logging
import os
from typing import Tuple

import numpy as np
import torch

torch.set_num_threads(1)

import flwr as fl
from accelerate.state import AcceleratorState
from datasets import Dataset
from datasets.utils import logging as datasets_logging
from flwr.common import Context, Metrics
from flwr.common.parameter import ndarrays_to_parameters
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import GenerationConfig
from transformers.utils import logging as transformers_logging

from fed_rag import RAGConfig, RAGSystem
from fed_rag.fl_tasks.huggingface import _get_weights
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.trainer_managers import HuggingFaceRAGTrainerManager
from fed_rag.trainers import HuggingFaceTrainerForLSR

from domain_aware_fedavg import DomainAwareFedAvg
from prepare_multi_domain_data import (
    DEFAULT_CLIENT_CONFIGS,
    MAX_CORPUS_DOCS,
    MAX_FINAL_TEST_PAIRS,
    MAX_SERVER_VAL_PAIRS,
    MAX_TRAIN_PAIRS,
    SEED,
    TOP_K,
    create_retriever,
    evaluate_retriever,
    setup_multi_domain_experiment,
)


NUM_ROUNDS = 4
BATCH_SIZE = 8
LEARNING_RATE = 2e-6
GENERATOR_MODEL = "distilgpt2"
TARGET_DATASET = "nfcorpus"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUTPUT_DIR, "output_csv_files")
LOG_DIR = os.path.join(OUTPUT_DIR, "output_log_files")
TMP_TRAINING_DIR = os.path.join(OUTPUT_DIR, ".tmp_training")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("federated_das")
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
datasets_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


CLIENT_KNOWLEDGE_STORES: dict[str, object] = {}
CLIENT_TRAIN_DATA: dict[str, list[dict]] = {}
CLIENT_DOMAINS: dict[str, str] = {}
TARGET_SERVER_VAL_PAIRS: list[dict] = []
TARGET_FINAL_TEST_PAIRS: list[dict] = []
TARGET_KNOWLEDGE_STORE = None
ROUND_METRICS: list[dict[str, float]] = []
TAU = 0.0
CURRENT_SEED = SEED
CURRENT_NUM_ROUNDS = NUM_ROUNDS
CURRENT_LOCAL_EPOCHS = 1


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
        return {"torch_dtype": _torch.float32, "device_map": "auto"}
    return {"torch_dtype": _torch.float32}


def get_client_resources() -> dict[str, float]:
    import torch as _torch

    if _torch.cuda.is_available():
        return {"num_cpus": 2, "num_gpus": 0.34}
    return {"num_cpus": 1, "num_gpus": 0}


def hash_weights(model) -> str:
    flat = np.concatenate(
        [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    )
    return hashlib.md5(flat.tobytes()).hexdigest()


def client_sort_key(cid: str):
    return (0, int(cid)) if cid.isdigit() else (1, cid)


def hash_jsonable(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def build_split_summary(client_data_map):
    summary = {}
    for cid in sorted(client_data_map.keys(), key=client_sort_key):
        pairs = client_data_map[cid]
        summary[cid] = {
            "num_pairs": len(pairs),
            "domain": CLIENT_DOMAINS.get(cid, "unknown"),
            "pairs_hash": hash_jsonable(pairs),
        }
    return summary


def build_csv_fieldnames(client_ids):
    base_fields = [
        "round",
        "tau",
        "seed",
        "target_dataset",
        "avg_loss",
        "aggregated_delta_norm",
        "aggregated_model_hash",
        "num_selected",
        "num_total",
        "best_round_so_far",
        "selected_best_round",
        "pre_server_val_mrr",
        "pre_server_val_recall_at_k",
        "pre_server_val_ndcg_at_k",
        "pre_final_test_mrr",
        "pre_final_test_recall_at_k",
        "pre_final_test_ndcg_at_k",
        "server_val_mrr",
        "server_val_recall_at_k",
        "server_val_ndcg_at_k",
        "final_test_mrr",
        "final_test_recall_at_k",
        "final_test_ndcg_at_k",
    ]
    for cid in client_ids:
        base_fields.extend(
            [
                f"client_{cid}_domain",
                f"client_{cid}_relevance",
                f"client_{cid}_selected",
                f"client_{cid}_loss",
                f"client_{cid}_num_examples",
                f"client_{cid}_size_weight",
                f"client_{cid}_delta_norm",
            ]
        )
    return base_fields


def build_run_slug(
    *, tau: float, seed: int, num_rounds: int, local_epochs: int, target: str
) -> str:
    target_slug = target.replace("-", "_")
    return (
        f"tau_{tau:.2f}_seed_{seed}_target_{target_slug}_"
        f"r{num_rounds}_e{local_epochs}"
    )


def set_retriever_weights(retriever, parameters) -> None:
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
            TARGET_KNOWLEDGE_STORE,
            TARGET_SERVER_VAL_PAIRS,
            top_k=TOP_K,
        )

    return evaluator


def evaluate_single_run_acceptance(
    round_infos,
    *,
    tau: float,
    relevance_scores: dict,
    target_client_id: str,
    min_selected: int,
):
    sorted_by_relevance = sorted(
        relevance_scores.items(), key=lambda item: item[1], reverse=True
    )

    report = {
        "tau_zero_selects_all": {
            "applicable": tau == 0.0,
            "passed": None,
        },
        "positive_tau_excludes_some_clients": {
            "applicable": tau > 0.0,
            "passed": None,
        },
        "target_client_always_selected": {
            "applicable": True,
            "passed": None,
        },
        "selection_matches_threshold_policy": {
            "applicable": True,
            "passed": None,
        },
        "relevance_scores_ordered": {
            "applicable": True,
            "passed": None,
        },
        "distinct_round_trajectories": {
            "applicable": True,
            "passed": None,
        },
    }

    if tau == 0.0:
        report["tau_zero_selects_all"]["passed"] = all(
            qi["num_selected"] == qi["num_total"] for qi in round_infos
        )

    # Skip round 1: proxy→logical CID mapping is not established until
    # after the first aggregate_fit, so selection cannot use tau.
    policy_round_infos = round_infos[1:] if len(round_infos) > 1 else round_infos

    if tau > 0.0:
        report["positive_tau_excludes_some_clients"]["passed"] = any(
            qi["num_selected"] < qi["num_total"] for qi in policy_round_infos
        )
        report["positive_tau_excludes_some_clients"]["evidence"] = {
            "excluded_per_round": [
                qi["num_total"] - qi["num_selected"] for qi in policy_round_infos
            ]
        }

    report["target_client_always_selected"]["passed"] = all(
        qi.get("selection_map", {}).get(target_client_id, False) for qi in round_infos
    )

    policy_matches = []
    for qi in policy_round_infos:
        above_tau = [
            cid for cid, rel in sorted_by_relevance if rel > tau
        ]
        if len(above_tau) >= min_selected:
            expected = set(above_tau)
        else:
            expected = {cid for cid, _ in sorted_by_relevance[:min_selected]}
        observed = set(qi.get("selected_cids", []))
        policy_matches.append(expected == observed)
    report["selection_matches_threshold_policy"]["passed"] = all(policy_matches)
    report["selection_matches_threshold_policy"]["evidence"] = {
        "per_round": policy_matches
    }

    report["relevance_scores_ordered"]["passed"] = all(
        sorted_by_relevance[i][1] >= sorted_by_relevance[i + 1][1]
        for i in range(len(sorted_by_relevance) - 1)
    )
    report["relevance_scores_ordered"]["evidence"] = {
        "scores": {k: f"{v:.4f}" for k, v in sorted_by_relevance}
    }

    unique_hashes = {qi["aggregated_model_hash"] for qi in round_infos}
    report["distinct_round_trajectories"]["passed"] = len(unique_hashes) == len(
        round_infos
    )
    report["distinct_round_trajectories"]["evidence"] = {
        "unique_hashes": len(unique_hashes),
        "rounds": len(round_infos),
    }

    return report


def client_fn(context: Context):
    cid = str(context.node_config["partition-id"])
    AcceleratorState._reset_state()

    logger.info(
        "Client %s (%s): Creating RAG system...",
        cid,
        CLIENT_DOMAINS.get(cid, "?"),
    )

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
        knowledge_store=CLIENT_KNOWLEDGE_STORES[cid],
        generator=generator,
        retriever=retriever,
        rag_config=RAGConfig(top_k=TOP_K),
    )

    data = CLIENT_TRAIN_DATA[cid]
    train_dataset = Dataset.from_dict(
        {
            "query": [p["query"] for p in data],
            "response": [p["response"] for p in data],
        }
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(
            TMP_TRAINING_DIR,
            f"seed_{CURRENT_SEED}",
            f"tau_{TAU:.2f}",
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
        metrics["logical_cid"] = cid
        return weights, num_examples, metrics

    flower_client.fit = audited_fit

    logger.info(
        "Client %s (%s): Ready (%s examples)",
        cid,
        CLIENT_DOMAINS.get(cid, "?"),
        len(train_dataset),
    )
    return flower_client.to_client()


def weighted_average(metrics: list[Tuple[int, Metrics]]) -> Metrics:
    losses = [num_examples * float(m["loss"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    avg_loss = (sum(losses) / total_examples) if total_examples > 0 else 0.0
    ROUND_METRICS.append({"avg_loss": avg_loss})
    return {"loss": avg_loss}


def main(
    tau: float,
    *,
    seed: int = SEED,
    num_rounds: int = NUM_ROUNDS,
    local_epochs: int = 1,
    target: str = TARGET_DATASET,
):
    global CLIENT_TRAIN_DATA, CLIENT_KNOWLEDGE_STORES, CLIENT_DOMAINS
    global TARGET_SERVER_VAL_PAIRS, TARGET_FINAL_TEST_PAIRS, TARGET_KNOWLEDGE_STORE
    global TAU, CURRENT_SEED, CURRENT_NUM_ROUNDS, CURRENT_LOCAL_EPOCHS

    TAU = tau
    CURRENT_SEED = seed
    CURRENT_NUM_ROUNDS = num_rounds
    CURRENT_LOCAL_EPOCHS = local_epochs

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TMP_TRAINING_DIR, exist_ok=True)

    run_slug = build_run_slug(
        tau=tau,
        seed=seed,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        target=target,
    )
    log_file = os.path.join(LOG_DIR, f"log_{run_slug}.log")
    csv_path = os.path.join(CSV_DIR, f"results_{run_slug}.csv")
    manifest_path = os.path.join(CSV_DIR, f"manifest_{run_slug}.json")
    acceptance_path = os.path.join(CSV_DIR, f"acceptance_{run_slug}.json")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)

    try:
        print("=" * 70)
        print(
            f"Start | τ={tau:.2f} | seed={seed} | "
            f"rounds={num_rounds} | epochs={local_epochs} | target={target}"
        )
        print(f"Device | runtime={get_runtime_device()}")
        print("=" * 70)

        experiment = setup_multi_domain_experiment(
            target_dataset=target,
            max_train=MAX_TRAIN_PAIRS,
            max_server_val=MAX_SERVER_VAL_PAIRS,
            max_final_test=MAX_FINAL_TEST_PAIRS,
            max_docs=MAX_CORPUS_DOCS,
            seed=seed,
        )

        client_data_map = experiment["client_data"]
        relevance_scores = experiment["relevance_scores"]
        target_cid = experiment["target_cid"]

        CLIENT_KNOWLEDGE_STORES = {}
        CLIENT_TRAIN_DATA = {}
        CLIENT_DOMAINS = {}
        for cid, data in client_data_map.items():
            CLIENT_KNOWLEDGE_STORES[cid] = data["knowledge_store"]
            CLIENT_TRAIN_DATA[cid] = data["train_pairs"]
            CLIENT_DOMAINS[cid] = data["label"]

        target_data = client_data_map[target_cid]
        TARGET_SERVER_VAL_PAIRS = target_data["server_val_pairs"]
        TARGET_FINAL_TEST_PAIRS = target_data["final_test_pairs"]
        TARGET_KNOWLEDGE_STORE = target_data["knowledge_store"]

        split_summary = build_split_summary(CLIENT_TRAIN_DATA)

        retriever = experiment["retriever"]
        pre_server_val_metrics = evaluate_retriever(
            retriever,
            TARGET_KNOWLEDGE_STORE,
            TARGET_SERVER_VAL_PAIRS,
            top_k=TOP_K,
        )
        pre_final_test_metrics = evaluate_retriever(
            retriever,
            TARGET_KNOWLEDGE_STORE,
            TARGET_FINAL_TEST_PAIRS,
            top_k=TOP_K,
        )
        print(
            f"\nPre server-val (target={target}) | "
            f"MRR={pre_server_val_metrics['mrr']:.4f} | "
            f"Recall@k={pre_server_val_metrics['recall_at_k']:.4f} | "
            f"NDCG@k={pre_server_val_metrics['ndcg_at_k']:.4f}"
        )
        print(
            f"Pre final-test (target={target}) | "
            f"MRR={pre_final_test_metrics['mrr']:.4f} | "
            f"Recall@k={pre_final_test_metrics['recall_at_k']:.4f} | "
            f"NDCG@k={pre_final_test_metrics['ndcg_at_k']:.4f}"
        )

        AcceleratorState._reset_state()
        model = retriever.query_encoder if retriever.query_encoder else retriever.encoder
        initial_hash = hash_weights(model)
        ndarrays = _get_weights(model)
        initial_parameters = ndarrays_to_parameters(ndarrays)
        print(f"Initial model hash: {initial_hash[:16]}")

        num_clients = len(client_data_map)
        strategy = DomainAwareFedAvg(
            tau=tau,
            client_relevance_scores=relevance_scores,
            min_selected=1,
            target_client_id=target_cid,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=1,
            min_available_clients=num_clients,
            fit_metrics_aggregation_fn=weighted_average,
            post_aggregation_evaluator=make_post_aggregation_evaluator(),
            initial_parameters=initial_parameters,
        )

        print("Running federated simulation...")
        ROUND_METRICS.clear()
        strategy.round_quality_info.clear()
        strategy.best_global_ndarrays = None
        strategy.best_round = None
        strategy.best_post_eval_metrics = {}

        client_resources = get_client_resources()
        print(
            f"Resources | client_cpus={client_resources['num_cpus']} | "
            f"client_gpus={client_resources['num_gpus']}"
        )

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

        best_ndarrays = strategy.best_global_ndarrays or strategy.last_global_ndarrays
        if best_ndarrays is None:
            raise RuntimeError("No aggregated model weights were produced.")

        best_round = strategy.best_round or len(strategy.round_quality_info)
        if strategy.best_post_eval_metrics:
            best_server_val_metrics = dict(strategy.best_post_eval_metrics)
        else:
            best_retriever = create_retriever()
            set_retriever_weights(best_retriever, best_ndarrays)
            best_server_val_metrics = evaluate_retriever(
                best_retriever,
                TARGET_KNOWLEDGE_STORE,
                TARGET_SERVER_VAL_PAIRS,
                top_k=TOP_K,
            )

        best_retriever = create_retriever()
        set_retriever_weights(best_retriever, best_ndarrays)
        final_test_metrics = evaluate_retriever(
            best_retriever,
            TARGET_KNOWLEDGE_STORE,
            TARGET_FINAL_TEST_PAIRS,
            top_k=TOP_K,
        )

        all_client_ids_for_csv = sorted(CLIENT_DOMAINS.keys(), key=client_sort_key)
        running_best_round = 0
        running_best_metrics: dict[str, float] = {}

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=build_csv_fieldnames(all_client_ids_for_csv)
            )
            writer.writeheader()

            for idx, qi in enumerate(strategy.round_quality_info):
                current_server_val = dict(qi.get("post_eval_metrics", {}))
                current_mrr = float(current_server_val.get("mrr", 0.0))
                running_mrr = float(running_best_metrics.get("mrr", 0.0))
                current_ndcg = float(current_server_val.get("ndcg_at_k", 0.0))
                running_ndcg = float(running_best_metrics.get("ndcg_at_k", 0.0))
                if (
                    not running_best_metrics
                    or current_mrr > running_mrr + 1e-12
                    or (
                        abs(current_mrr - running_mrr) <= 1e-12
                        and current_ndcg > running_ndcg + 1e-12
                    )
                ):
                    running_best_round = qi["round"]
                    running_best_metrics = current_server_val

                participating_records = {
                    record["cid"]: record for record in qi["client_records"]
                }
                selection_map = qi.get("selection_map", {})
                row = {
                    "round": qi["round"],
                    "tau": f"{tau:.2f}",
                    "seed": str(seed),
                    "target_dataset": target,
                    "avg_loss": f"{qi['aggregated_loss']:.6f}",
                    "aggregated_delta_norm": f"{qi['aggregated_delta_norm']:.6f}",
                    "aggregated_model_hash": qi["aggregated_model_hash"],
                    "num_selected": qi["num_selected"],
                    "num_total": qi["num_total"],
                    "best_round_so_far": str(running_best_round),
                    "selected_best_round": str(best_round),
                    "pre_server_val_mrr": f"{pre_server_val_metrics['mrr']:.6f}",
                    "pre_server_val_recall_at_k": f"{pre_server_val_metrics['recall_at_k']:.6f}",
                    "pre_server_val_ndcg_at_k": f"{pre_server_val_metrics['ndcg_at_k']:.6f}",
                    "pre_final_test_mrr": f"{pre_final_test_metrics['mrr']:.6f}",
                    "pre_final_test_recall_at_k": f"{pre_final_test_metrics['recall_at_k']:.6f}",
                    "pre_final_test_ndcg_at_k": f"{pre_final_test_metrics['ndcg_at_k']:.6f}",
                    "server_val_mrr": f"{current_server_val.get('mrr', 0.0):.6f}",
                    "server_val_recall_at_k": f"{current_server_val.get('recall_at_k', 0.0):.6f}",
                    "server_val_ndcg_at_k": f"{current_server_val.get('ndcg_at_k', 0.0):.6f}",
                    "final_test_mrr": f"{final_test_metrics.get('mrr', 0.0):.6f}",
                    "final_test_recall_at_k": f"{final_test_metrics.get('recall_at_k', 0.0):.6f}",
                    "final_test_ndcg_at_k": f"{final_test_metrics.get('ndcg_at_k', 0.0):.6f}",
                }

                if idx < len(ROUND_METRICS):
                    row["avg_loss"] = f"{ROUND_METRICS[idx]['avg_loss']:.6f}"

                for cid in all_client_ids_for_csv:
                    row[f"client_{cid}_domain"] = CLIENT_DOMAINS.get(cid, "")
                    row[f"client_{cid}_relevance"] = f"{relevance_scores.get(cid, 0.0):.4f}"
                    row[f"client_{cid}_selected"] = (
                        "1" if selection_map.get(cid, False) else "0"
                    )

                    record = participating_records.get(cid)
                    if record is not None:
                        row[f"client_{cid}_loss"] = f"{record['loss']:.6f}"
                        row[f"client_{cid}_num_examples"] = str(record["num_examples"])
                        row[f"client_{cid}_size_weight"] = f"{record['size_weight']:.6f}"
                        row[f"client_{cid}_delta_norm"] = f"{record['delta_norm']:.6f}"
                    else:
                        row[f"client_{cid}_loss"] = ""
                        row[f"client_{cid}_num_examples"] = ""
                        row[f"client_{cid}_size_weight"] = ""
                        row[f"client_{cid}_delta_norm"] = ""

                writer.writerow(row)

        acceptance_report = evaluate_single_run_acceptance(
            strategy.round_quality_info,
            tau=tau,
            relevance_scores=relevance_scores,
            target_client_id=target_cid,
            min_selected=1,
        )
        with open(acceptance_path, "w", encoding="utf-8") as f:
            json.dump(acceptance_report, f, indent=2, sort_keys=True)

        manifest = {
            "experiment_name": "das_fedavg_domain_selection",
            "canonical_results_csv": csv_path,
            "canonical_log_file": log_file,
            "acceptance_report": acceptance_path,
            "tau": tau,
            "seed": seed,
            "target_dataset": target,
            "target_client_id": target_cid,
            "client_domains": dict(CLIENT_DOMAINS),
            "relevance_scores": {k: float(v) for k, v in relevance_scores.items()},
            "num_rounds": num_rounds,
            "num_clients": len(CLIENT_DOMAINS),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "local_epochs": local_epochs,
            "max_train_pairs": MAX_TRAIN_PAIRS,
            "max_server_val_pairs": MAX_SERVER_VAL_PAIRS,
            "max_final_test_pairs": MAX_FINAL_TEST_PAIRS,
            "max_corpus_docs": MAX_CORPUS_DOCS,
            "top_k": TOP_K,
            "initial_model_hash": initial_hash,
            "pre_server_val_metrics": pre_server_val_metrics,
            "pre_final_test_metrics": pre_final_test_metrics,
            "best_round": best_round,
            "best_server_val_metrics": best_server_val_metrics,
            "final_test_metrics": final_test_metrics,
            "client_split_summary": split_summary,
            "csv_schema_version": 2,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        print("\n" + "=" * 70)
        print(f"DAS-FEDAVG COMPLETE | τ={tau:.2f} | best_round={best_round}")
        print(
            "Best server-val | "
            f"MRR={best_server_val_metrics.get('mrr', 0.0):.4f} | "
            f"NDCG@k={best_server_val_metrics.get('ndcg_at_k', 0.0):.4f}"
        )
        print(
            "Final test      | "
            f"MRR={final_test_metrics.get('mrr', 0.0):.4f} | "
            f"Recall@k={final_test_metrics.get('recall_at_k', 0.0):.4f} | "
            f"NDCG@k={final_test_metrics.get('ndcg_at_k', 0.0):.4f}"
        )
        print(f"Results   : {csv_path}")
        print(f"Manifest  : {manifest_path}")
        print(f"Acceptance: {acceptance_path}")
        print("=" * 70)
        return csv_path
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAS-FedAvg domain-aware experiment"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="Domain relevance threshold (0 = all clients, higher = more selective)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--rounds", type=int, default=NUM_ROUNDS, help="Number of federated rounds"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local training epochs per client per round",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_DATASET,
        help="Target domain BEIR dataset name",
    )
    args = parser.parse_args()

    main(
        tau=args.tau,
        seed=args.seed,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        target=args.target,
    )
