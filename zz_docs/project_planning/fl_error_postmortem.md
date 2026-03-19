# Flower-Based Federated Training — Walkthrough

## Changes Made

### 1. Fixed `loss=0` Bug

#### [huggingface.py](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/src/fed_rag/trainer_managers/huggingface.py)

```diff:huggingface.py
"""HuggingFace RAG Trainer"""

from typing import TYPE_CHECKING, Any, Callable, cast

from typing_extensions import assert_never

from fed_rag.base.trainer import BaseTrainer
from fed_rag.base.trainer_manager import BaseRAGTrainerManager, RAGTrainMode
from fed_rag.data_structures.results import TestResult, TrainResult
from fed_rag.decorators import federate
from fed_rag.exceptions import (
    MissingExtraError,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
)

try:
    from datasets import Dataset

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer

    from fed_rag.fl_tasks.huggingface import HFModelType, HuggingFaceFLTask


class HuggingFaceRAGTrainerManager(BaseRAGTrainerManager):
    """HuggingFace RAG Trainer Manager"""

    def __init__(
        self,
        mode: RAGTrainMode,
        retriever_trainer: BaseTrainer | None = None,
        generator_trainer: BaseTrainer | None = None,
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)
        super().__init__(
            mode=mode,
            retriever_trainer=retriever_trainer,
            generator_trainer=generator_trainer,
            **kwargs,
        )

    def _prepare_generator_for_training(self, **kwargs: Any) -> None:
        self.generator_trainer.model.train()

        # freeze generator
        if self.retriever_trainer:
            self.retriever_trainer.model.eval()

    def _prepare_retriever_for_training(
        self, freeze_context_encoder: bool = True, **kwargs: Any
    ) -> None:
        self.retriever_trainer.model.train()

        # freeze generator
        if self.generator_trainer:
            self.generator_trainer.model.eval()

    def _train_retriever(self, **kwargs: Any) -> TrainResult:
        if self.retriever_trainer:
            self._prepare_retriever_for_training()
            return self.retriever_trainer.train(**kwargs)
        else:
            raise UnspecifiedRetrieverTrainer(
                "Attempted to perform retriever trainer with an unspecified trainer."
            )

    def _train_generator(self, **kwargs: Any) -> TrainResult:
        if self.generator_trainer:
            self._prepare_generator_for_training()
            return self.generator_trainer.train(**kwargs)
        else:
            raise UnspecifiedGeneratorTrainer(
                "Attempted to perform generator trainer with an unspecified trainer."
            )

    def train(self, **kwargs: Any) -> TrainResult:
        if self.mode == "retriever":
            return self._train_retriever(**kwargs)
        elif self.mode == "generator":
            return self._train_generator(**kwargs)
        else:
            assert_never(self.mode)  # pragma: no cover

    def _get_federated_trainer(self) -> tuple[Callable, "HFModelType"]:
        if self.mode == "retriever":
            if self.retriever_trainer is None:
                raise UnspecifiedRetrieverTrainer(
                    "Cannot federate an unspecified retriever trainer."
                )
            retriever_train_fn = self.retriever_trainer.train
            retriever_module = self.retriever_trainer.model
            retriever_module = cast("SentenceTransformer", retriever_module)

            # Create a standalone function for federation
            def train_wrapper(
                model: "HFModelType",
                train_dataset: "Dataset",
                val_dataset: "Dataset",
            ) -> TrainResult:
                _ = retriever_train_fn()
                return TrainResult(loss=0)

            return (
                federate.trainer.huggingface(train_wrapper),
                retriever_module,
            )

        elif self.mode == "generator":
            if self.generator_trainer is None:
                raise UnspecifiedGeneratorTrainer(
                    "Cannot federate an unspecified generator trainer."
                )
            generator_train_fn = self.generator_trainer.train
            generator_module = self.generator_trainer.model

            # Create a standalone function for federation
            def train_wrapper(
                model: "HFModelType",  # TODO: handle union types in inspector
                train_dataset: "Dataset",
                val_dataset: "Dataset",
            ) -> TrainResult:
                _ = generator_train_fn()
                # TODO get loss from out
                return TrainResult(loss=0)

            return (
                federate.trainer.huggingface(train_wrapper),
                generator_module,
            )
        else:
            assert_never(self.mode)  # pragma: no cover

    def get_federated_task(self) -> "HuggingFaceFLTask":
        from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask

        federated_trainer, _module = self._get_federated_trainer()

        # TODO: add logic for getting evaluator/tester and then federate it as well
        # federated_tester = self.get_federated_tester(tester_decorator)
        # For now, using a simple placeholder test function
        def test_fn(
            model: "HFModelType", eval_dataset: "Dataset"
        ) -> TestResult:
            # Implement simple testing or return a placeholder
            return TestResult(loss=0.42, metrics={})  # pragma: no cover

        federated_tester = federate.tester.huggingface(test_fn)

        return HuggingFaceFLTask.from_trainer_and_tester(
            trainer=federated_trainer,
            tester=federated_tester,
        )
===
"""HuggingFace RAG Trainer"""

from typing import TYPE_CHECKING, Any, Callable, cast

from typing_extensions import assert_never

from fed_rag.base.trainer import BaseTrainer
from fed_rag.base.trainer_manager import BaseRAGTrainerManager, RAGTrainMode
from fed_rag.data_structures.results import TestResult, TrainResult
from fed_rag.decorators import federate
from fed_rag.exceptions import (
    MissingExtraError,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
)

try:
    from datasets import Dataset

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer

    from fed_rag.fl_tasks.huggingface import HFModelType, HuggingFaceFLTask


class HuggingFaceRAGTrainerManager(BaseRAGTrainerManager):
    """HuggingFace RAG Trainer Manager"""

    def __init__(
        self,
        mode: RAGTrainMode,
        retriever_trainer: BaseTrainer | None = None,
        generator_trainer: BaseTrainer | None = None,
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)
        super().__init__(
            mode=mode,
            retriever_trainer=retriever_trainer,
            generator_trainer=generator_trainer,
            **kwargs,
        )

    def _prepare_generator_for_training(self, **kwargs: Any) -> None:
        self.generator_trainer.model.train()

        # freeze generator
        if self.retriever_trainer:
            self.retriever_trainer.model.eval()

    def _prepare_retriever_for_training(
        self, freeze_context_encoder: bool = True, **kwargs: Any
    ) -> None:
        self.retriever_trainer.model.train()

        # freeze generator
        if self.generator_trainer:
            self.generator_trainer.model.eval()

    def _train_retriever(self, **kwargs: Any) -> TrainResult:
        if self.retriever_trainer:
            self._prepare_retriever_for_training()
            return self.retriever_trainer.train(**kwargs)
        else:
            raise UnspecifiedRetrieverTrainer(
                "Attempted to perform retriever trainer with an unspecified trainer."
            )

    def _train_generator(self, **kwargs: Any) -> TrainResult:
        if self.generator_trainer:
            self._prepare_generator_for_training()
            return self.generator_trainer.train(**kwargs)
        else:
            raise UnspecifiedGeneratorTrainer(
                "Attempted to perform generator trainer with an unspecified trainer."
            )

    def train(self, **kwargs: Any) -> TrainResult:
        if self.mode == "retriever":
            return self._train_retriever(**kwargs)
        elif self.mode == "generator":
            return self._train_generator(**kwargs)
        else:
            assert_never(self.mode)  # pragma: no cover

    def _get_federated_trainer(self) -> tuple[Callable, "HFModelType"]:
        if self.mode == "retriever":
            if self.retriever_trainer is None:
                raise UnspecifiedRetrieverTrainer(
                    "Cannot federate an unspecified retriever trainer."
                )
            retriever_train_fn = self.retriever_trainer.train
            retriever_module = self.retriever_trainer.model
            retriever_module = cast("SentenceTransformer", retriever_module)

            # Create a standalone function for federation
            def train_wrapper(
                model: "HFModelType",
                train_dataset: "Dataset",
                val_dataset: "Dataset",
            ) -> TrainResult:
                result = retriever_train_fn()
                return TrainResult(loss=result.loss)

            return (
                federate.trainer.huggingface(train_wrapper),
                retriever_module,
            )

        elif self.mode == "generator":
            if self.generator_trainer is None:
                raise UnspecifiedGeneratorTrainer(
                    "Cannot federate an unspecified generator trainer."
                )
            generator_train_fn = self.generator_trainer.train
            generator_module = self.generator_trainer.model

            # Create a standalone function for federation
            def train_wrapper(
                model: "HFModelType",  # TODO: handle union types in inspector
                train_dataset: "Dataset",
                val_dataset: "Dataset",
            ) -> TrainResult:
                result = generator_train_fn()
                return TrainResult(loss=result.loss)

            return (
                federate.trainer.huggingface(train_wrapper),
                generator_module,
            )
        else:
            assert_never(self.mode)  # pragma: no cover

    def get_federated_task(self) -> "HuggingFaceFLTask":
        from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask

        federated_trainer, _module = self._get_federated_trainer()

        # TODO: add logic for getting evaluator/tester and then federate it as well
        # federated_tester = self.get_federated_tester(tester_decorator)
        # For now, using a simple placeholder test function
        def test_fn(
            model: "HFModelType", eval_dataset: "Dataset"
        ) -> TestResult:
            # Implement simple testing or return a placeholder
            return TestResult(loss=0.42, metrics={})  # pragma: no cover

        federated_tester = federate.tester.huggingface(test_fn)

        return HuggingFaceFLTask.from_trainer_and_tester(
            trainer=federated_trainer,
            tester=federated_tester,
        )
```

Both [train_wrapper](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/src/fed_rag/trainer_managers/huggingface.py#130-137) functions in [_get_federated_trainer()](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/src/fed_rag/trainer_managers/huggingface.py#97-144) now capture and forward the actual training loss instead of hardcoding `loss=0`.

### 2. Updated Tests

#### [test_hf_trainer_manager.py](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/tests/trainer_managers/huggingface/test_hf_trainer_manager.py)

```diff:test_hf_trainer_manager.py
import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from pytest import MonkeyPatch

from fed_rag.base.trainer import (
    BaseGeneratorTrainer,
    BaseRetrieverTrainer,
    BaseTrainer,
)
from fed_rag.base.trainer_manager import BaseRAGTrainerManager, RAGTrainMode
from fed_rag.exceptions import (
    MissingExtraError,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
    UnsupportedTrainerMode,
)
from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HuggingFaceRAGTrainerManager.__mro__
    ]
    assert BaseRAGTrainerManager.__name__ in names_of_base_classes


def test_init(
    retriever_trainer: BaseRetrieverTrainer,
    generator_trainer: BaseGeneratorTrainer,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        train_dataset=train_dataset,
        retriever_trainer=retriever_trainer,
        generator_trainer=generator_trainer,
    )

    assert manager.generator_trainer == generator_trainer
    assert manager.retriever_trainer == retriever_trainer
    assert manager.mode == RAGTrainMode.RETRIEVER


def test_init_raises_unspecified_generator_trainer_error(
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Generator trainer must be set when in generator mode",
    ):
        HuggingFaceRAGTrainerManager(
            mode="generator",
        )


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_retriever_for_training")
def test_init_raises_unspecified_retriever_trainer_error(
    mock_prepare_retriever_for_training: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Retriever trainer must be set when in retriever mode",
    ):
        HuggingFaceRAGTrainerManager(
            mode="retriever",
        )


def test_huggingface_extra_missing(
    retriever_trainer: BaseRetrieverTrainer,
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.trainer_managers.huggingface"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceRAGTrainerManager` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.trainer_managers.huggingface import (
                HuggingFaceRAGTrainerManager,
            )

            HuggingFaceRAGTrainerManager(
                mode="retriever",
                retriever_trainer=retriever_trainer,
                generator_trainer=generator_trainer,
            )

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_retriever_for_training")
def test_train_retriever(
    mock_prepare_retriever_for_training: MagicMock,
    retriever_trainer: BaseRetrieverTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    mock_retriever_trainer = MagicMock()
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )
    manager.retriever_trainer = mock_retriever_trainer

    manager.train()

    mock_prepare_retriever_for_training.assert_called_once()
    mock_retriever_trainer.train.assert_called_once_with()


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_generator_for_training")
def test_train_generator(
    mock_prepare_generator_for_training: MagicMock,
    generator_trainer: BaseGeneratorTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )
    mock_generator_trainer = MagicMock()
    manager.generator_trainer = mock_generator_trainer

    manager.train()

    mock_prepare_generator_for_training.assert_called_once()
    mock_generator_trainer.train.assert_called_once_with()


def test_get_federated_task_retriever(
    retriever_trainer: BaseTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )

    # act
    retriever_trainer, _ = trainer._get_federated_trainer()
    out = retriever_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert fl_task._trainer_spec == retriever_trainer.__fl_task_trainer_config


def test_get_federated_task_generator(
    generator_trainer: BaseTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )

    # act
    generator_trainer, _ = trainer._get_federated_trainer()
    out = generator_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert fl_task._trainer_spec == generator_trainer.__fl_task_trainer_config


def test_invalid_mode_raises_error(
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    msg = (
        f"Unsupported RAG train mode: both. "
        f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
    )
    with pytest.raises(UnsupportedTrainerMode, match=msg):
        HuggingFaceRAGTrainerManager(
            mode="both",
        )


def test_get_federated_task_raises_unspecified_generator_error(
    monkeypatch: MonkeyPatch,
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    # this will pass validations at init
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever", retriever_trainer=retriever_trainer
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Cannot federate an unspecified generator trainer.",
    ):
        manager.mode = "generator"  # user modifies the mode
        manager.get_federated_task()


def test_private_get_federated_task_raises_unspecified_retriever_error(
    monkeypatch: MonkeyPatch,
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    # this will pass validations at init
    manager = HuggingFaceRAGTrainerManager(
        mode="generator", generator_trainer=generator_trainer
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Cannot federate an unspecified retriever trainer.",
    ):
        # change mode to retriever
        manager.mode = "retriever"
        manager.get_federated_task()


def test_prepare_generator_for_training(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")
    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
        retriever_trainer=retriever_trainer,
    )

    # add mocks
    mock_generator_model = MagicMock()
    mock_retriever_model = MagicMock()
    manager.generator_trainer.model = mock_generator_model
    manager.retriever_trainer.model = mock_retriever_model

    # Check parameters before
    for param in generator_trainer.model.parameters():
        assert param.requires_grad is True  # They should start unfrozen

    # act
    with does_not_raise():
        manager._prepare_generator_for_training()

    # assert
    mock_generator_model.train.assert_called_once()
    mock_retriever_model.eval.assert_called_once()
    for param in generator_trainer.model.parameters():
        assert param.requires_grad is False  # They should now be frozen


def test_prepare_retriever_for_training(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        generator_trainer=generator_trainer,
        retriever_trainer=retriever_trainer,
    )

    # add mocks
    mock_generator_model = MagicMock()
    mock_retriever_model = MagicMock()
    manager.generator_trainer.model = mock_generator_model
    manager.retriever_trainer.model = mock_retriever_model

    # Check parameters before
    for param in retriever_trainer.model.parameters():
        assert param.requires_grad is True  # They should start unfrozen

    # act
    with does_not_raise():
        manager._prepare_retriever_for_training()

    # assert
    mock_generator_model.eval.assert_called_once()
    mock_retriever_model.train.assert_called_once()
    for param in retriever_trainer.model.parameters():
        assert param.requires_grad is False  # They should now be frozen


def test_private_train_retriever_raises_unspecified_retriever_error(
    generator_trainer: BaseGeneratorTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # no retriever trainer set
    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Attempted to perform retriever trainer with an unspecified trainer.",
    ):
        manager._train_retriever()


def test_private_train_generator_raises_unspecified_generator_error(
    retriever_trainer: BaseGeneratorTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # no retriever trainer set
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Attempted to perform generator trainer with an unspecified trainer.",
    ):
        manager._train_generator()
===
import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from pytest import MonkeyPatch

from fed_rag.base.trainer import (
    BaseGeneratorTrainer,
    BaseRetrieverTrainer,
    BaseTrainer,
)
from fed_rag.base.trainer_manager import BaseRAGTrainerManager, RAGTrainMode
from fed_rag.exceptions import (
    MissingExtraError,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
    UnsupportedTrainerMode,
)
from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HuggingFaceRAGTrainerManager.__mro__
    ]
    assert BaseRAGTrainerManager.__name__ in names_of_base_classes


def test_init(
    retriever_trainer: BaseRetrieverTrainer,
    generator_trainer: BaseGeneratorTrainer,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        train_dataset=train_dataset,
        retriever_trainer=retriever_trainer,
        generator_trainer=generator_trainer,
    )

    assert manager.generator_trainer == generator_trainer
    assert manager.retriever_trainer == retriever_trainer
    assert manager.mode == RAGTrainMode.RETRIEVER


def test_init_raises_unspecified_generator_trainer_error(
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Generator trainer must be set when in generator mode",
    ):
        HuggingFaceRAGTrainerManager(
            mode="generator",
        )


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_retriever_for_training")
def test_init_raises_unspecified_retriever_trainer_error(
    mock_prepare_retriever_for_training: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Retriever trainer must be set when in retriever mode",
    ):
        HuggingFaceRAGTrainerManager(
            mode="retriever",
        )


def test_huggingface_extra_missing(
    retriever_trainer: BaseRetrieverTrainer,
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.trainer_managers.huggingface"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceRAGTrainerManager` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.trainer_managers.huggingface import (
                HuggingFaceRAGTrainerManager,
            )

            HuggingFaceRAGTrainerManager(
                mode="retriever",
                retriever_trainer=retriever_trainer,
                generator_trainer=generator_trainer,
            )

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_retriever_for_training")
def test_train_retriever(
    mock_prepare_retriever_for_training: MagicMock,
    retriever_trainer: BaseRetrieverTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    mock_retriever_trainer = MagicMock()
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )
    manager.retriever_trainer = mock_retriever_trainer

    manager.train()

    mock_prepare_retriever_for_training.assert_called_once()
    mock_retriever_trainer.train.assert_called_once_with()


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_generator_for_training")
def test_train_generator(
    mock_prepare_generator_for_training: MagicMock,
    generator_trainer: BaseGeneratorTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )
    mock_generator_trainer = MagicMock()
    manager.generator_trainer = mock_generator_trainer

    manager.train()

    mock_prepare_generator_for_training.assert_called_once()
    mock_generator_trainer.train.assert_called_once_with()


def test_get_federated_task_retriever(
    retriever_trainer: BaseTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )

    # act
    retriever_trainer, _ = trainer._get_federated_trainer()
    out = retriever_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert — train_wrapper now returns actual loss from trainer.train()
    # TestRetrieverTrainer.train() returns TrainResult(loss=0.42)
    assert out.loss == 0.42
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert fl_task._trainer_spec == retriever_trainer.__fl_task_trainer_config


def test_get_federated_task_generator(
    generator_trainer: BaseTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )

    # act
    generator_trainer, _ = trainer._get_federated_trainer()
    out = generator_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert — train_wrapper now returns actual loss from trainer.train()
    # TestGeneratorTrainer.train() returns TrainResult(loss=0.42)
    assert out.loss == 0.42
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert fl_task._trainer_spec == generator_trainer.__fl_task_trainer_config


def test_invalid_mode_raises_error(
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    msg = (
        f"Unsupported RAG train mode: both. "
        f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
    )
    with pytest.raises(UnsupportedTrainerMode, match=msg):
        HuggingFaceRAGTrainerManager(
            mode="both",
        )


def test_get_federated_task_raises_unspecified_generator_error(
    monkeypatch: MonkeyPatch,
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    # this will pass validations at init
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever", retriever_trainer=retriever_trainer
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Cannot federate an unspecified generator trainer.",
    ):
        manager.mode = "generator"  # user modifies the mode
        manager.get_federated_task()


def test_private_get_federated_task_raises_unspecified_retriever_error(
    monkeypatch: MonkeyPatch,
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    # this will pass validations at init
    manager = HuggingFaceRAGTrainerManager(
        mode="generator", generator_trainer=generator_trainer
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Cannot federate an unspecified retriever trainer.",
    ):
        # change mode to retriever
        manager.mode = "retriever"
        manager.get_federated_task()


def test_prepare_generator_for_training(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")
    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
        retriever_trainer=retriever_trainer,
    )

    # add mocks
    mock_generator_model = MagicMock()
    mock_retriever_model = MagicMock()
    manager.generator_trainer.model = mock_generator_model
    manager.retriever_trainer.model = mock_retriever_model

    # Check parameters before
    for param in generator_trainer.model.parameters():
        assert param.requires_grad is True  # They should start unfrozen

    # act
    with does_not_raise():
        manager._prepare_generator_for_training()

    # assert
    mock_generator_model.train.assert_called_once()
    mock_retriever_model.eval.assert_called_once()
    for param in generator_trainer.model.parameters():
        assert param.requires_grad is False  # They should now be frozen


def test_prepare_retriever_for_training(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        generator_trainer=generator_trainer,
        retriever_trainer=retriever_trainer,
    )

    # add mocks
    mock_generator_model = MagicMock()
    mock_retriever_model = MagicMock()
    manager.generator_trainer.model = mock_generator_model
    manager.retriever_trainer.model = mock_retriever_model

    # Check parameters before
    for param in retriever_trainer.model.parameters():
        assert param.requires_grad is True  # They should start unfrozen

    # act
    with does_not_raise():
        manager._prepare_retriever_for_training()

    # assert
    mock_generator_model.eval.assert_called_once()
    mock_retriever_model.train.assert_called_once()
    for param in retriever_trainer.model.parameters():
        assert param.requires_grad is False  # They should now be frozen


def test_private_train_retriever_raises_unspecified_retriever_error(
    generator_trainer: BaseGeneratorTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # no retriever trainer set
    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Attempted to perform retriever trainer with an unspecified trainer.",
    ):
        manager._train_retriever()


def test_private_train_generator_raises_unspecified_generator_error(
    retriever_trainer: BaseGeneratorTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # no retriever trainer set
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Attempted to perform generator trainer with an unspecified trainer.",
    ):
        manager._train_generator()
```

Tests now assert `loss == 0.42` (the value from `TestRetrieverTrainer.train()` / `TestGeneratorTrainer.train()` fixtures) instead of `loss == 0`.

### 3. New Script

#### [federated_lsr_flwr.py](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/baseline/federated_lsr_flwr.py)

Uses `fl.simulation.start_simulation()` with:
- [client_fn(cid)](file:///Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/baseline/federated_lsr_flwr.py#87-158) creating fresh RAG systems per client
- `initial_parameters` provided to `FedAvg` (avoids Ray serialization mismatch)
- `fit_metrics_aggregation_fn` to aggregate and display per-round losses

---

## Verification Results

### Tests: 16/16 pass

```
tests/trainer_managers/huggingface/test_hf_trainer_manager.py::test_get_federated_task_retriever PASSED
tests/trainer_managers/huggingface/test_hf_trainer_manager.py::test_get_federated_task_generator PASSED
======================== 16 passed, 1 warning in 0.14s =======================
```

### Flower Simulation: 3 rounds, loss decreasing

```
[ROUND 1] aggregate_fit: received 2 results and 0 failures
[ROUND 2] aggregate_fit: received 2 results and 0 failures
[ROUND 3] aggregate_fit: received 2 results and 0 failures

History (metrics, distributed, fit):
  loss: [(1, 0.0279), (2, 0.0093), (3, 0.0036)]
```

### Dependency Installed

```
pip install "flwr[simulation]"  →  ray-2.51.1, msgpack-1.1.2
```
