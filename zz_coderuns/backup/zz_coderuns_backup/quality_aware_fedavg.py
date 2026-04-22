import hashlib
import numpy as np
from typing import Any, Callable, Optional, Union

from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy


class QualityAwareFedAvg(FedAvg):

    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 1e-8,
        focus_client_id: Optional[str] = None,
        post_aggregation_evaluator: Optional[
            Callable[[int, list[np.ndarray]], dict[str, float]]
        ] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.focus_client_id = focus_client_id
        self.post_aggregation_evaluator = post_aggregation_evaluator
        self.round_quality_info: list[dict[str, Any]] = []
        self.last_global_ndarrays: Optional[list[np.ndarray]] = None

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
            logical_cid = str(
                fit_res.metrics.get("logical_cid", client_proxy.cid)
            )
            client_data.append(
                {
                    "cid": logical_cid,
                    "ndarrays": ndarrays,
                    "n_examples": n_examples,
                    "loss": float(loss),
                    "fit_metrics": dict(fit_res.metrics),
                }
            )

        client_data.sort(key=lambda item: self._client_sort_key(item["cid"]))

        losses = np.array([item["loss"] for item in client_data], dtype=float)
        inverse_losses = 1.0 / (losses + self.epsilon)
        quality_scores = inverse_losses / inverse_losses.sum()

        n_examples = np.array(
            [item["n_examples"] for item in client_data], dtype=float
        )
        size_weights = n_examples / n_examples.sum()

        combined_weights = (
            self.alpha * quality_scores
            + (1 - self.alpha) * size_weights
        )
        combined_weights = combined_weights / combined_weights.sum()

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
                layer_sum += combined_weights[j] * client_weights[layer_idx]
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

        client_records = []
        for idx, item in enumerate(client_data):
            client_records.append(
                {
                    "cid": item["cid"],
                    "loss": float(losses[idx]),
                    "num_examples": int(item["n_examples"]),
                    "quality_score": float(quality_scores[idx]),
                    "size_weight": float(size_weights[idx]),
                    "combined_weight": float(combined_weights[idx]),
                    "delta_norm": float(client_delta_norms[idx]),
                    "loss_source": str(
                        item["fit_metrics"].get("loss_source", "unknown")
                    ),
                    "loss_stage": str(
                        item["fit_metrics"].get("loss_stage", "unknown")
                    ),
                }
            )

        post_eval_metrics: dict[str, float] = {}
        if self.post_aggregation_evaluator is not None:
            post_eval_metrics = self.post_aggregation_evaluator(
                server_round, aggregated
            )

        info = {
            "round": server_round,
            "alpha": self.alpha,
            "aggregated_loss": float(metrics_aggregated.get("loss", 0.0)),
            "aggregated_delta_norm": aggregated_delta_norm,
            "aggregated_model_hash": self._hash_ndarrays(aggregated),
            "post_eval_metrics": post_eval_metrics,
            "client_records": client_records,
        }
        self.round_quality_info.append(info)

        focus_record = None
        if self.focus_client_id is not None:
            focus_record = next(
                (
                    record for record in client_records
                    if record["cid"] == self.focus_client_id
                ),
                None,
            )
        if focus_record is None and client_records:
            focus_record = max(
                client_records, key=lambda record: record["loss"]
            )

        summary = (
            f"  Round {server_round} | α={self.alpha:.1f} | "
            f"avg_loss={info['aggregated_loss']:.6f} | "
            f"global_delta={aggregated_delta_norm:.6f}"
        )
        if focus_record is not None:
            summary += (
                f" | focus_client={focus_record['cid']}"
                f" loss={focus_record['loss']:.6f}"
                f" weight={focus_record['combined_weight']:.4f}"
            )
        if post_eval_metrics:
            summary += (
                f" | post_mrr={post_eval_metrics.get('mrr', 0.0):.4f}"
                f" | post_ndcg={post_eval_metrics.get('ndcg_at_k', 0.0):.4f}"
            )
        print(summary, flush=True)

        self.last_global_ndarrays = aggregated

        return parameters_aggregated, metrics_aggregated
