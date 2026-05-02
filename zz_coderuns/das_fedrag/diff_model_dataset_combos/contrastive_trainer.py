"""Contrastive retriever trainer — replaces distilgpt2-based LSR.

Replaces the LSR (KL-divergence against distilgpt2 scores) local training
objective with MultipleNegativesRankingLoss (InfoNCE).  Training signal comes
directly from (query, positive_doc) pairs; no LM teacher is involved.

Why this matters for QA-FedAvg:
  - With LSR + distilgpt2, every client produced byte-for-byte identical model
    updates regardless of whether their data was clean or corrupted, because
    distilgpt2 assigns uniform likelihood to all medical text.
  - With contrastive training, a noisy client whose 'response' is a random
    document pushes the query encoder toward an unrelated document → measurably
    destructive gradient → different model update → quality signal discriminates.

Drop-in usage in client_fn:
    model = retriever.query_encoder
    contrastive_client = ContrastiveFlowerClient(
        model=model,
        train_dataset=train_dataset,   # "query" + "response" columns
        training_args=training_args,
    )
    original_fit = contrastive_client.fit
    contrastive_client.fit = audited_fit   # monkey-patch as before
    return contrastive_client.to_client()
"""

import torch
from typing import Dict, Tuple

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments

import flwr as fl
from flwr.common import NDArrays


# ---------------------------------------------------------------------------
# Weight helpers (consistent ordering with set_retriever_weights in main script)
# ---------------------------------------------------------------------------

def _get_weights(model: SentenceTransformer) -> NDArrays:
    """Return state_dict values as a list of numpy arrays."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def _set_weights(model: SentenceTransformer, parameters: NDArrays) -> None:
    """Load a list of numpy arrays back into the model state_dict."""
    state_dict = model.state_dict()
    updated = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
    model.load_state_dict(updated, strict=True)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContrastiveRetrieverTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer using MultipleNegativesRankingLoss.

    Overrides create_optimizer for the same reason LSRSentenceTransformerTrainer
    does: MNR loss has no trainable parameters, so the parent's default
    optimizer builder (which inspects loss.parameters()) would produce an empty
    optimizer.  This override always uses model.named_parameters() instead.
    """

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            decay_params = self.get_decay_parameter_names(self.model)
            grouped = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if n in decay_params and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if n not in decay_params and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                grouped,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer


# ---------------------------------------------------------------------------
# Flower-compatible client
# ---------------------------------------------------------------------------

class ContrastiveFlowerClient:
    """Drop-in replacement for the LSR flower_client produced by HuggingFaceRAGTrainerManager.

    Holds a reference to the SentenceTransformer model (which must be the SAME
    Python object as retriever.query_encoder so that audited_fit's quality signal
    sees post-training weights on the retriever).

    Interface mirrors what fl_task.client() previously returned:
      .fit(parameters: NDArrays, config: dict) -> (NDArrays, int, dict)
      .to_client() -> fl.client.Client
    """

    def __init__(
        self,
        model: SentenceTransformer,
        train_dataset: Dataset,
        training_args: SentenceTransformerTrainingArguments,
    ):
        self.model = model
        # MNR expects "anchor" (query) and "positive" (relevant document) columns.
        self.train_dataset = train_dataset.rename_columns(
            {"query": "anchor", "response": "positive"}
        )
        self.training_args = training_args
        # Loss is created once; it holds no trainable state.
        self.loss = MultipleNegativesRankingLoss(model)

    def fit(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[NDArrays, int, Dict]:
        # Receive global weights from the server.
        _set_weights(self.model, parameters)

        # Local contrastive training.
        trainer = ContrastiveRetrieverTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            loss=self.loss,
        )
        output = trainer.train()

        updated_weights = _get_weights(self.model)
        num_examples = len(self.train_dataset)
        metrics = {"loss": float(output.training_loss)}
        return updated_weights, num_examples, metrics

    def to_client(self) -> fl.client.Client:
        """Wrap in a proper Flower Client (compatible with fl.simulation)."""
        outer = self

        class _NumpyClient(fl.client.NumPyClient):
            def fit(self, parameters, config):
                # By the time to_client() is called, outer.fit may have been
                # monkey-patched to audited_fit — call whatever .fit is now.
                return outer.fit(parameters, config)

            def evaluate(self, parameters, config):
                return 0.0, 0, {}

        return _NumpyClient().to_client()
