"""
Experiment Translator: Convert text hypotheses to experiment configurations.

Takes a natural language hypothesis about vision models and outputs a validated
experiment configuration JSON that maps to available models/datasets.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class UntranslatableHypothesis(Exception):
    """Raised when hypothesis cannot be translated to experiment config."""

    pass


# Available resources from experiment/src/
AVAILABLE_MODELS = {
    "deit_small": "DeiT-Small",
    "resnet34": "ResNet-34",
}

# Reverse mapping for display names
MODEL_DISPLAY_TO_KEY = {v: k for k, v in AVAILABLE_MODELS.items()}
MODEL_DISPLAY_TO_KEY.update(
    {
        "deit-small": "deit_small",
        "resnet-34": "resnet34",
    }
)

AVAILABLE_DATASETS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "stl10": "STL-10",
    "flowers102": "Flowers102",
    "oxford_pets": "Oxford-Pets",
}

DATASET_DISPLAY_TO_KEY = {v: k for k, v in AVAILABLE_DATASETS.items()}
DATASET_DISPLAY_TO_KEY.update(
    {
        "cifar-10": "cifar10",
        "cifar-100": "cifar100",
        "stl-10": "stl10",
    }
)

AVAILABLE_TRAINING_MODES = ["linear_probe", "knn_evaluate", "train_from_scratch"]

# Constraints from task requirements
MAX_EPOCHS = 5
MAX_MODELS = 2
MAX_DATASETS = 2


class ExperimentTranslator:
    """
    Translate HypothesisSpec to experiment configurations.

    Performs deterministic mapping from structured hypothesis specifications
    to experiment configurations. Validates all outputs against available
    resources and applies small-scale constraints.
    """

    def __init__(self):
        """Initialize translator."""
        logger.info("Initialized ExperimentTranslator with deterministic mapping")

    async def translate(self, spec: dict) -> dict[str, Any]:
        """
        Translate hypothesis spec to experiment configuration.

        Args:
            spec: HypothesisSpec dictionary with required fields:
                - task: str (e.g., "image_classification")
                - dataset: str (e.g., "flowers102")
                - baseline: dict with "model" key
                - variant: dict with "model" key
                - metric: str (e.g., "top1_accuracy")

        Returns:
            Validated experiment config dict with keys:
            - models: list of model names (1-2)
            - datasets: list of dataset names (1-2)
            - training_mode: "linear_probe", "knn_evaluate", or "train_from_scratch"
            - epochs: int (1-5)
            - batch_size: int
            - learning_rate: float
            - metrics_to_track: list of metrics

        Raises:
            UntranslatableHypothesis: If required fields are missing or invalid
        """
        logger.info(f"Translating hypothesis spec: {spec.get('hypothesis', '')[:100]}...")

        required_fields = ["task", "dataset", "baseline", "variant", "metric"]
        missing = [f for f in required_fields if f not in spec]
        if missing:
            raise UntranslatableHypothesis(f"Missing required fields: {missing}")

        baseline_model = spec["baseline"].get("model")
        variant_model = spec["variant"].get("model")
        if not baseline_model or not variant_model:
            raise UntranslatableHypothesis("Both baseline and variant must have 'model' field")

        baseline_normalized = self._normalize_model_name(baseline_model)
        variant_normalized = self._normalize_model_name(variant_model)
        if not baseline_normalized:
            raise UntranslatableHypothesis(
                f"Invalid baseline model '{baseline_model}'. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        if not variant_normalized:
            raise UntranslatableHypothesis(
                f"Invalid variant model '{variant_model}'. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        dataset_normalized = self._normalize_dataset_name(spec["dataset"])
        if not dataset_normalized:
            raise UntranslatableHypothesis(
                f"Invalid dataset '{spec['dataset']}'. Available datasets: {list(AVAILABLE_DATASETS.keys())}"
            )

        config = {
            "models": [AVAILABLE_MODELS[baseline_normalized], AVAILABLE_MODELS[variant_normalized]],
            "datasets": [AVAILABLE_DATASETS[dataset_normalized]],
            "training_mode": "linear_probe",
            "epochs": 5,
            "batch_size": 128,
            "learning_rate": 0.001,
            "metrics_to_track": [spec["metric"], "loss"],
        }

        validated_config = self._validate_and_constrain(config)

        logger.info(f"Final config: {json.dumps(validated_config, indent=2)}")
        return validated_config

    def _validate_and_constrain(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        """
        Validate config against available resources and apply constraints.

        Raises UntranslatableHypothesis if validation fails.
        """
        validated = {}

        # Validate models
        models = raw_config.get("models", [])
        validated_models = []
        for model in models[:MAX_MODELS]:
            # Normalize model name
            model_key = self._normalize_model_name(model)
            if model_key:
                validated_models.append(AVAILABLE_MODELS[model_key])

        if not validated_models:
            raise UntranslatableHypothesis(
                f"No valid models found in {models}. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        validated["models"] = validated_models[:MAX_MODELS]

        # Validate datasets
        datasets = raw_config.get("datasets", [])
        validated_datasets = []
        for dataset in datasets[:MAX_DATASETS]:
            dataset_key = self._normalize_dataset_name(dataset)
            if dataset_key:
                validated_datasets.append(AVAILABLE_DATASETS[dataset_key])

        if not validated_datasets:
            raise UntranslatableHypothesis(
                f"No valid datasets found in {datasets}. Available datasets: {list(AVAILABLE_DATASETS.keys())}"
            )

        validated["datasets"] = validated_datasets[:MAX_DATASETS]

        # Validate training mode
        training_mode = raw_config.get("training_mode", "linear_probe")
        if training_mode not in AVAILABLE_TRAINING_MODES:
            logger.warning(f"Invalid training mode '{training_mode}'. Using linear_probe.")
            training_mode = "linear_probe"
        validated["training_mode"] = training_mode

        # Constrain epochs
        epochs = raw_config.get("epochs", 3)
        try:
            epochs = int(epochs)
            epochs = max(1, min(epochs, MAX_EPOCHS))
        except (ValueError, TypeError):
            logger.warning(f"Invalid epochs '{epochs}'. Using 3.")
            epochs = 3
        validated["epochs"] = epochs

        # Validate batch size
        batch_size = raw_config.get("batch_size", 128)
        try:
            batch_size = int(batch_size)
            batch_size = max(16, min(batch_size, 512))  # Reasonable range
        except (ValueError, TypeError):
            batch_size = 128
        validated["batch_size"] = batch_size

        # Validate learning rate
        learning_rate = raw_config.get("learning_rate", 0.001)
        try:
            learning_rate = float(learning_rate)
            learning_rate = max(0.0001, min(learning_rate, 0.1))
        except (ValueError, TypeError):
            learning_rate = 0.001
        validated["learning_rate"] = learning_rate

        # Metrics
        metrics = raw_config.get("metrics_to_track", ["accuracy", "loss"])
        if not isinstance(metrics, list) or not metrics:
            metrics = ["accuracy", "loss"]
        validated["metrics_to_track"] = metrics

        return validated

    def _normalize_model_name(self, model: Any) -> Optional[str]:
        """Normalize model name to internal key."""
        if not isinstance(model, str):
            return None

        model_clean = model.lower().strip().replace(" ", "").replace("-", "")

        # Direct match
        if model_clean in AVAILABLE_MODELS:
            return model_clean

        # Display name match
        model_display = model.strip()
        if model_display in MODEL_DISPLAY_TO_KEY:
            return MODEL_DISPLAY_TO_KEY[model_display]

        # Fuzzy match
        if "deit" in model_clean or "vit" in model_clean:
            return "deit_small"
        if "resnet" in model_clean:
            return "resnet34"

        return None

    def _normalize_dataset_name(self, dataset: Any) -> Optional[str]:
        """Normalize dataset name to internal key."""
        if not isinstance(dataset, str):
            return None

        dataset_clean = dataset.lower().strip().replace(" ", "").replace("-", "")

        # Direct match
        if dataset_clean in AVAILABLE_DATASETS:
            return dataset_clean

        # Display name match
        dataset_display = dataset.strip()
        if dataset_display in DATASET_DISPLAY_TO_KEY:
            return DATASET_DISPLAY_TO_KEY[dataset_display]

        # Fuzzy match
        if "cifar10" in dataset_clean:
            return "cifar10"
        if "cifar100" in dataset_clean:
            return "cifar100"
        if "stl" in dataset_clean:
            return "stl10"
        if "flower" in dataset_clean:
            return "flowers102"
        if "pet" in dataset_clean or "oxford" in dataset_clean:
            return "oxford_pets"

        return None


async def main():
    """Example usage."""
    translator = ExperimentTranslator()

    hypothesis_spec = {
        "hypothesis": "DeiT-Small may achieve better transfer learning than ResNet-34 on fine-grained tasks",
        "confidence": 0.75,
        "task": "image_classification",
        "dataset": "flowers102",
        "baseline": {"model": "resnet34", "pretrain": "imagenet"},
        "variant": {"model": "deit_small", "pretrain": "imagenet"},
        "metric": "top1_accuracy",
        "expected_effect": {"direction": "increase", "min_delta_points": 2},
    }

    config = await translator.translate(hypothesis_spec)
    print("Generated experiment configuration:")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
