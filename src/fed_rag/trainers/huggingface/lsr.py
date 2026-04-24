"""HuggingFace LM-Supervised Retriever Trainer"""

from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import PrivateAttr, model_validator

from fed_rag import RAGSystem
from fed_rag.base.trainer import BaseRetrieverTrainer
from fed_rag.data_collators.huggingface import DataCollatorForLSR
from fed_rag.data_structures.results import TestResult, TrainResult
from fed_rag.exceptions import (
    InvalidDataCollatorError,
    InvalidLossError,
    MissingExtraError,
    MissingInputTensor,
    TrainerError,
)
from fed_rag.loss.pytorch.lsr import LSRLoss
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin
from fed_rag.utils.huggingface import _validate_rag_system

try:
    from sentence_transformers import SentenceTransformerTrainer

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

    class SentenceTransformerTrainer:  # type: ignore[no-redef]
        """Dummy placeholder when sentence transformers is not available."""

        pass


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
    )
    from transformers import TrainingArguments
    from transformers.trainer_utils import TrainOutput


class LSRSentenceTransformerTrainer(SentenceTransformerTrainer):
    def __init__(
        self,
        *args: Any,
        data_collator: DataCollatorForLSR,
        loss: Optional[LSRLoss] = None,
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        # set loss
        if loss is None:
            loss = LSRLoss()
        else:
            if not isinstance(loss, LSRLoss):
                raise InvalidLossError(
                    "`LSRSentenceTransformerTrainer` must use ~fed_rag.loss.LSRLoss`."
                )

        if not isinstance(data_collator, DataCollatorForLSR):
            raise InvalidDataCollatorError(
                "`LSRSentenceTransformerTrainer` must use ~fed_rag.data_collators.DataCollatorForLSR`."
            )

        super().__init__(
            *args, loss=loss, data_collator=data_collator, **kwargs
        )

    def compute_loss(
        self,
        model: "SentenceTransformer",
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute LSR loss.

        The differentiable forward pass through the encoder happens HERE
        (not in the data collator) to avoid issues with DataLoader prefetching.
        """
        queries = inputs["queries"]
        context_texts_batch = inputs["context_texts"]
        lm_scores = inputs["lm_scores"]

        # If model is wrapped in DataParallel (e.g. multi-gpu environment), extract underlying module
        raw_model = model.module if hasattr(model, "module") else model

        batch_retriever_scores = []
        for query, context_texts in zip(queries, context_texts_batch):
            # query embedding via model forward (preserves grad graph)
            query_features = raw_model.tokenize([query])
            query_features = {
                k: v.to(model.device if hasattr(model, "device") else raw_model.device) for k, v in query_features.items()
            }
            query_embedding = model(query_features)["sentence_embedding"]

            # context embeddings (no grad needed)
            with torch.no_grad():
                context_embedding = raw_model.encode(
                    context_texts, convert_to_tensor=True
                )

            # cosine similarity (differentiable w.r.t. query_embedding)
            query_norm = torch.nn.functional.normalize(
                query_embedding, p=2, dim=1
            )
            context_norm = torch.nn.functional.normalize(
                context_embedding, p=2, dim=1
            )
            retriever_scores = torch.mm(
                query_norm, context_norm.t()
            ).squeeze(0)
            batch_retriever_scores.append(retriever_scores)

        retrieval_scores = torch.stack(batch_retriever_scores, dim=0)
        loss = self.loss(retrieval_scores, lm_scores)

        return (loss, inputs) if return_outputs else loss

    
    def create_optimizer(self) -> "torch.optim.Optimizer":
        """Override to ensure the retriever model's params are optimized.

        The parent SentenceTransformerTrainer builds optimizer param groups
        from the loss module's parameters. Since LSRLoss has no trainable
        params, the optimizer ends up empty. This override uses the actual
        model parameters instead.
        """
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = torch.optim.AdamW, {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
        return self.optimizer



class HuggingFaceTrainerForLSR(HuggingFaceTrainerMixin, BaseRetrieverTrainer):
    """HuggingFace LM-Supervised Retriever Trainer."""

    _hf_trainer: Optional["SentenceTransformerTrainer"] = PrivateAttr(
        default=None
    )

    def __init__(
        self,
        rag_system: RAGSystem,
        train_dataset: "Dataset",
        training_arguments: Optional["TrainingArguments"] = None,
        **kwargs: Any,
    ):
        super().__init__(
            train_dataset=train_dataset,
            rag_system=rag_system,
            training_arguments=training_arguments,
            **kwargs,
        )

    @model_validator(mode="after")
    def set_private_attributes(self) -> "HuggingFaceTrainerForLSR":
        # if made it to here, then this import is available
        from sentence_transformers import SentenceTransformer

        # validate rag system
        _validate_rag_system(self.rag_system)

        # validate model
        if not isinstance(self.model, SentenceTransformer):
            raise TrainerError(
                "For `HuggingFaceTrainerForLSR`, attribute `model` must be of type "
                "`~sentence_transformers.SentenceTransformer`."
            )

        self._hf_trainer = LSRSentenceTransformerTrainer(
            model=self.model,
            args=self.training_arguments,
            data_collator=DataCollatorForLSR(rag_system=self.rag_system),
            train_dataset=self.train_dataset,
        )

        return self

    def train(self, **kwargs: Any) -> TrainResult:
        output: TrainOutput = self.hf_trainer_obj.train(**kwargs)
        return TrainResult(loss=output.training_loss)

    def evaluate(self) -> TestResult:
        # TODO: implement this
        raise NotImplementedError

    @property
    def hf_trainer_obj(self) -> "SentenceTransformerTrainer":
        return self._hf_trainer
