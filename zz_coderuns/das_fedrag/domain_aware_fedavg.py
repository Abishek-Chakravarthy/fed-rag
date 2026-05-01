import hashlib
import numpy as np
from typing import Any, Callable, Optional, Union

from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy


class DomainAwareFedAvg(FedAvg):

    def __init__(
        self,
        tau: float = 1.0,
        client_relevance_scores: Optional[dict[str, float]] = None,
        min_selected: int = 1,
        target_client_id: Optional[str] = None,
        pre_registered_cid_map: Optional[dict[str, str]] = None,
        post_aggregation_evaluator: Optional[
            Callable[[int, list[np.ndarray]], dict[str, float]]
        ] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.client_relevance_scores = client_relevance_scores or {}
        self.min_selected = min_selected
        self.target_client_id = target_client_id
        self.post_aggregation_evaluator = post_aggregation_evaluator
        self.round_quality_info: list[dict[str, Any]] = []
        self.last_global_ndarrays: Optional[list[np.ndarray]] = None
        self.best_global_ndarrays: Optional[list[np.ndarray]] = None
        self.best_round: Optional[int] = None
        self.best_post_eval_metrics: dict[str, float] = {}
        self.last_selection_map: dict[str, bool] = {}
        self.last_selected_cids: list[str] = []

        # Pre-register proxy→logical CID mapping so Round 1 applies correct
        # domain-aware weighting from the start (not a select-all fallback).
        if pre_registered_cid_map is not None:
            self.proxy_to_logical_cid: dict[str, str] = dict(pre_registered_cid_map)
        else:
            self.proxy_to_logical_cid = {}

        if self.initial_parameters is not None:
            self.last_global_ndarrays = parameters_to_ndarrays(
                self.initial_parameters
            )

    @staticmethod
    def _client_sort_key(cid: str) -> tuple[int, Union[int, str]]:
        if cid.isdigit():
            return (0, int(cid))
        return (1, cid)

    @staticmethod
    def _hash_ndarrays(ndarrays: list[np.ndarray]) -> str:
        digest = hashlib.md5()
        for arr in ndarrays:
            digest.update(arr.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _is_better_eval(
        candidate: dict[str, float], incumbent: dict[str, float]
    ) -> bool:
        if not candidate:
            return False
        if not incumbent:
            return True
        candidate_mrr = float(candidate.get("mrr", 0.0))
        incumbent_mrr = float(incumbent.get("mrr", 0.0))
        if candidate_mrr > incumbent_mrr + 1e-12:
            return True
        if abs(candidate_mrr - incumbent_mrr) <= 1e-12:
            return float(candidate.get("ndcg_at_k", 0.0)) > float(
                incumbent.get("ndcg_at_k", 0.0)
            ) + 1e-12
        return False

    def _resolve_clients(
        self, available_cids: list[str], server_round: int
    ) -> tuple[list[str], dict[str, bool], dict[str, str]]:
        """Resolve proxy CIDs to logical CIDs. All clients participate every round.

        Returns:
            selected_cids: all available proxy CIDs (soft weighting — no exclusion)
            selection_map: proxy_cid → True for all (everyone participates)
            cid_aliases:   proxy_cid → logical_cid display label
        """
        cid_aliases = {}
        for cid in available_cids:
            logical_cid = self.proxy_to_logical_cid.get(cid)
            if logical_cid is None:
                print(
                    f"  WARNING: Round {server_round} — proxy CID '{cid}' has no "
                    "pre-registered logical CID mapping. Selecting anyway to discover."
                )
                cid_aliases[cid] = cid
            else:
                cid_aliases[cid] = logical_cid

        # All available clients participate to report their logical CIDs
        selected = available_cids
        selection_map = {cid: True for cid in available_cids}
        return selected, selection_map, cid_aliases

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        num_available = client_manager.num_available()
        all_clients = client_manager.sample(
            num_clients=num_available,
            min_num_clients=self.min_fit_clients,
        )

        available_cids = [client.cid for client in all_clients]
        selected_cids, selection_map, cid_aliases = self._resolve_clients(
            available_cids, server_round
        )
        self.last_selection_map = dict(selection_map)
        self.last_selected_cids = [cid_aliases.get(cid, cid) for cid in selected_cids]

        client_by_cid = {client.cid: client for client in all_clients}
        result = [
            (client_by_cid[cid], fit_ins)
            for cid in selected_cids
            if cid in client_by_cid
        ]

        # Log per-client relevance scores for diagnostics.
        selected_info = []
        for cid in sorted(available_cids, key=self._client_sort_key):
            logical_cid = cid_aliases.get(cid, cid)
            rel = self.client_relevance_scores.get(logical_cid, 0.0)
            sel = "✅" if selection_map.get(cid, False) else "❌"
            selected_info.append(
                f"    Client {logical_cid} (proxy={cid}): d={rel:.4f} {sel}"
            )

        print(
            f"\n  Round {server_round} | soft-domain-weighting | "
            f"Selected {len(result)}/{len(all_clients)} clients"
        )
        for line in selected_info:
            print(line)

        return result

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[
            Union[tuple[ClientProxy, FitRes], BaseException]
        ],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        client_data = []
        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            n_examples = fit_res.num_examples
            loss = fit_res.metrics.get("loss", 1.0)
            reported_logical_cid = str(
                fit_res.metrics.get("logical_cid", client_proxy.cid)
            )
            pre_registered = self.proxy_to_logical_cid.get(client_proxy.cid)
            if pre_registered is not None and pre_registered != reported_logical_cid:
                print(
                    f"  WARNING: Round {server_round} — proxy '{client_proxy.cid}' "
                    f"pre-registered as logical '{pre_registered}' but reported "
                    f"'{reported_logical_cid}'. Using pre-registered value."
                )
                logical_cid = pre_registered
            else:
                # Handles the no-pre-registration fallback gracefully.
                self.proxy_to_logical_cid[client_proxy.cid] = reported_logical_cid
                logical_cid = reported_logical_cid

            client_data.append({
                "cid": logical_cid,
                "ndarrays": ndarrays,
                "n_examples": n_examples,
                "loss": float(loss),
                "fit_metrics": dict(fit_res.metrics),
            })

        client_data.sort(key=lambda item: self._client_sort_key(item["cid"]))
        client_data = [item for item in client_data if item["n_examples"] > 0]
        if not client_data:
            if self.last_global_ndarrays is None:
                return None, {}
            return ndarrays_to_parameters(self.last_global_ndarrays), {"loss": 0.0}

        # --- Soft domain weighting -------------------------------------------
        import math

        n_total = sum(item["n_examples"] for item in client_data)
        
        log_relevances = []
        for item in client_data:
            d_j = self.client_relevance_scores.get(item["cid"], 0.0)
            log_relevances.append(math.log(max(d_j, 1e-8)))
            
        max_log = max(log_relevances)
        tau = self.tau if self.tau > 0 else 1.0  # safe fallback
        exp_scores = [math.exp((lr - max_log) / tau) for lr in log_relevances]
        softmax_sum = sum(exp_scores)
        softmax_weights = [e / softmax_sum for e in exp_scores]

        raw_weights = []
        for i, item in enumerate(client_data):
            raw_weights.append(softmax_weights[i] * (item["n_examples"] / n_total))

        raw_weight_sum = sum(raw_weights)
        if raw_weight_sum < 1e-12:
            # Degenerate case: all weights are ~0. Fall back to
            # uniform size-based weights so aggregation does not produce NaN.
            print(
                f"  WARNING: Round {server_round} — all raw domain weights are ~0. "
                "Falling back to uniform size-based aggregation."
            )
            n_examples_arr = np.array(
                [item["n_examples"] for item in client_data], dtype=float
            )
            final_weights = (n_examples_arr / n_examples_arr.sum()).tolist()
        else:
            final_weights = [w / raw_weight_sum for w in raw_weights]
        # ---------------------------------------------------------------------

        all_weights = [item["ndarrays"] for item in client_data]
        num_layers = len(all_weights[0])

        client_delta_norms = []
        if self.last_global_ndarrays is None:
            client_delta_norms = [0.0 for _ in client_data]
        else:
            for client_weights in all_weights:
                squared_sum = 0.0
                for prev_layer, client_layer in zip(
                    self.last_global_ndarrays, client_weights
                ):
                    diff = client_layer - prev_layer
                    squared_sum += float(np.sum(diff * diff))
                client_delta_norms.append(float(np.sqrt(squared_sum)))

        aggregated = []
        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(all_weights[0][layer_idx])
            for j, client_weights in enumerate(all_weights):
                layer_sum += final_weights[j] * client_weights[layer_idx]
            aggregated.append(layer_sum)

        parameters_aggregated = ndarrays_to_parameters(aggregated)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in results
            ]
            user_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics_aggregated.update(user_metrics)

        aggregated_delta_norm = 0.0
        if self.last_global_ndarrays is not None:
            squared_sum = 0.0
            for prev_layer, new_layer in zip(
                self.last_global_ndarrays, aggregated
            ):
                diff = new_layer - prev_layer
                squared_sum += float(np.sum(diff * diff))
            aggregated_delta_norm = float(np.sqrt(squared_sum))

        losses = np.array([item["loss"] for item in client_data], dtype=float)

        client_records = []
        for idx, item in enumerate(client_data):
            d_j = self.client_relevance_scores.get(item["cid"], 0.0)
            client_records.append({
                "cid": item["cid"],
                "loss": float(losses[idx]),
                "num_examples": int(item["n_examples"]),
                "domain_weight": float(final_weights[idx]),
                "raw_domain_weight": float(raw_weights[idx]),
                "delta_norm": float(client_delta_norms[idx]),
                "relevance": d_j,
                "selected": True,
            })

        post_eval_metrics: dict[str, float] = {}
        if self.post_aggregation_evaluator is not None:
            post_eval_metrics = self.post_aggregation_evaluator(
                server_round, aggregated
            )

        logical_selection_map = {
            self.proxy_to_logical_cid.get(proxy_cid, proxy_cid): selected
            for proxy_cid, selected in self.last_selection_map.items()
        }

        info = {
            "round": server_round,
            "tau": self.tau,
            "aggregated_loss": float(metrics_aggregated.get("loss", 0.0)),
            "aggregated_delta_norm": aggregated_delta_norm,
            "aggregated_model_hash": self._hash_ndarrays(aggregated),
            "post_eval_metrics": post_eval_metrics,
            "client_records": client_records,
            "num_selected": len(client_data),
            "num_total": len(self.client_relevance_scores),
            "selected_cids": [cid for cid, selected in logical_selection_map.items() if selected],
            "selection_map": logical_selection_map,
        }
        self.round_quality_info.append(info)

        if self._is_better_eval(post_eval_metrics, self.best_post_eval_metrics):
            self.best_post_eval_metrics = dict(post_eval_metrics)
            self.best_global_ndarrays = [arr.copy() for arr in aggregated]
            self.best_round = server_round

        # Log per-client domain weights for diagnostics.
        weight_summary = " | ".join(
            f"C{rec['cid']}:d={rec['relevance']:.3f}→w={rec['domain_weight']:.3f}"
            for rec in client_records
        )
        summary = (
            f"  Round {server_round} | soft-domain | "
            f"avg_loss={info['aggregated_loss']:.6f} | "
            f"global_delta={aggregated_delta_norm:.6f} | "
            f"clients={len(results)}/{len(self.client_relevance_scores)}"
        )
        if post_eval_metrics:
            summary += (
                f" | post_mrr={post_eval_metrics.get('mrr', 0.0):.4f}"
                f" | post_ndcg={post_eval_metrics.get('ndcg_at_k', 0.0):.4f}"
            )
        if self.best_round is not None:
            summary += f" | best_round={self.best_round}"
        print(summary, flush=True)
        print(f"  Weights | {weight_summary}", flush=True)

        self.last_global_ndarrays = aggregated

        return parameters_aggregated, metrics_aggregated