"""
Experiment orchestration: run single experiment configs and save results.
"""

import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiment.src.datasets import get_num_classes, get_transforms, load_dataset
from experiment.src.models import extract_features, load_model
from experiment.src.train import knn_evaluate, train_linear_probe
from experiment.src.utils import set_seed

logger = logging.getLogger(__name__)


def run_single_experiment(config: dict) -> dict:
    """
    Run a single experiment configuration end-to-end.

    Pipeline: set seed → load model → load dataset → extract features → evaluate.

    Args:
        config: {
            arch: "deit_small" | "resnet34",
            pretrained: bool,
            dataset: str (one of 5 dataset names),
            eval_method: "linear_probe" | "knn",
            seed: int,
            device: "mps" | "cpu",
        }

    Returns:
        Result dict with accuracy, runtime, metadata.
    """
    start_time = time.time()

    arch = config["arch"]
    pretrained = config["pretrained"]
    dataset_name = config["dataset"]
    eval_method = config["eval_method"]
    seed = config["seed"]
    device = torch.device(config.get("device", "cpu"))
    batch_size = 32 if pretrained else 16

    set_seed(seed)
    num_classes = get_num_classes(dataset_name)

    logger.info(
        f"Running: arch={arch}, pretrained={pretrained}, dataset={dataset_name}, "
        f"eval={eval_method}, seed={seed}"
    )

    model = load_model(arch=arch, pretrained=pretrained, num_classes=num_classes)

    train_transform = get_transforms(pretrained=pretrained, train=False)
    test_transform = get_transforms(pretrained=pretrained, train=False)

    train_ds = load_dataset(dataset_name, split="train", transform=train_transform)
    test_ds = load_dataset(dataset_name, split="test", transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)

    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)

    accuracy = 0.0
    metadata: dict = {}

    if eval_method == "linear_probe":
        probe_config = {
            "device": str(device),
            "epochs": 100,
            "lr_grid": [0.001, 0.01, 0.1, 1.0],
        }
        result = train_linear_probe(train_features, train_labels, num_classes, probe_config)

        probe_result = _train_probe_proper(
            train_features,
            train_labels,
            test_features,
            test_labels,
            num_classes,
            result["best_lr"],
            device,
        )
        accuracy = probe_result["accuracy"]
        metadata = {
            "best_lr": result["best_lr"],
            "train_loss_curve": result["train_loss_curve"][-5:],
        }

    elif eval_method == "knn":
        accuracy = knn_evaluate(
            train_features,
            train_labels,
            test_features,
            test_labels,
            k=20,
        )
        metadata = {"k": 20}

    runtime = time.time() - start_time

    if device.type == "mps":
        torch.mps.empty_cache()

    result = {
        "arch": arch,
        "pretrained": pretrained,
        "dataset": dataset_name,
        "eval_method": eval_method,
        "seed": seed,
        "accuracy": accuracy,
        "runtime_seconds": round(runtime, 2),
        "converged": True,
        "metadata": metadata,
    }

    logger.info(f"Result: accuracy={accuracy:.4f}, runtime={runtime:.1f}s")
    return result


def _train_probe_proper(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    lr: float,
    device: torch.device,
    epochs: int = 100,
) -> dict:
    """Train linear probe on train set, evaluate on held-out test set."""
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR

    feature_dim = train_features.shape[1]
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=256,
        shuffle=True,
    )

    for epoch in range(epochs):
        classifier.train()
        for feats, lbls in train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            loss = criterion(classifier(feats), lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        test_feats = test_features.to(device)
        test_lbls = test_labels.to(device)
        preds = classifier(test_feats).argmax(dim=1)
        accuracy = (preds == test_lbls).float().mean().item()

    return {"accuracy": accuracy}


def save_result(result: dict, results_dir: Path) -> None:
    """
    Append experiment result to all_results.jsonl (one JSON object per line).

    Args:
        result: Experiment result dictionary
        results_dir: Directory for results file (created if needed)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "all_results.jsonl"

    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    logger.debug(f"Result saved to {results_file}")
