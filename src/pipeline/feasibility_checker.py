"""Feasibility checker for validating hypotheses against available capabilities.

This module provides validation functions to ensure that hypothesis specifications
are feasible given the available experimental capabilities (models, datasets, metrics).

The feasibility checker prevents hypothesis-experiment mismatches by validating
that all required resources are available before hypothesis review and experiment
execution.
"""

from typing import Dict, List, Tuple, Any


def check_hypothesis_feasible(spec: Dict[str, Any], caps: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a hypothesis specification against available capabilities.

    Checks that all fields in the hypothesis spec (task, dataset, models, metric)
    are supported by the capabilities dictionary. Collects all validation errors
    rather than failing on the first error.

    Args:
        spec: Hypothesis specification dictionary with keys:
            - task: Task type (e.g., "image_classification")
            - dataset: Dataset name (e.g., "flowers102")
            - baseline: Dict with "model" key
            - variant: Dict with "model" key
            - metric: Evaluation metric (e.g., "top1_accuracy")
        caps: Capabilities dictionary with keys:
            - task: Supported task type
            - datasets: List of available datasets
            - models: List of available models
            - metrics: List of available metrics

    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
            - is_valid: True if all validations pass, False otherwise
            - error_messages: List of validation error messages (empty if valid)

    Examples:
        >>> caps = {
        ...     "task": "image_classification",
        ...     "models": ["deit_small", "resnet34"],
        ...     "datasets": ["cifar10", "flowers102"],
        ...     "metrics": ["top1_accuracy", "loss"],
        ... }
        >>> valid_spec = {
        ...     "task": "image_classification",
        ...     "dataset": "flowers102",
        ...     "baseline": {"model": "resnet34"},
        ...     "variant": {"model": "deit_small"},
        ...     "metric": "top1_accuracy",
        ... }
        >>> check_hypothesis_feasible(valid_spec, caps)
        (True, [])

        >>> invalid_spec = {
        ...     "task": "text_classification",
        ...     "dataset": "pubmed",
        ...     "baseline": {"model": "bert"},
        ...     "variant": {"model": "deit_small"},
        ...     "metric": "f1_score",
        ... }
        >>> is_valid, errors = check_hypothesis_feasible(invalid_spec, caps)
        >>> is_valid
        False
        >>> len(errors) > 0
        True
    """
    errors: List[str] = []

    # Validate task
    if spec.get("task") != caps.get("task"):
        errors.append(
            f"Task '{spec.get('task')}' not supported. Only '{caps.get('task')}' is available."
        )

    # Validate dataset
    dataset = spec.get("dataset")
    available_datasets = caps.get("datasets", [])
    if dataset not in available_datasets:
        errors.append(
            f"Dataset '{dataset}' not available. Available datasets: {available_datasets}"
        )

    # Validate baseline model
    baseline_model = spec.get("baseline", {}).get("model")
    available_models = caps.get("models", [])
    if baseline_model not in available_models:
        errors.append(
            f"Baseline model '{baseline_model}' not available. Available models: {available_models}"
        )

    # Validate variant model
    variant_model = spec.get("variant", {}).get("model")
    if variant_model not in available_models:
        errors.append(
            f"Variant model '{variant_model}' not available. Available models: {available_models}"
        )

    # Validate metric
    metric = spec.get("metric")
    available_metrics = caps.get("metrics", [])
    if metric not in available_metrics:
        errors.append(f"Metric '{metric}' not available. Available metrics: {available_metrics}")

    return (len(errors) == 0, errors)
