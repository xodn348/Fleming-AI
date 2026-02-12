"""
Training routines: linear probing, k-NN evaluation, and from-scratch training.

Linear probe: SGD with LR grid search on frozen features.
k-NN: Cosine similarity with majority vote.
From-scratch: AdamW (ViT) / SGD (ResNet) with warmup, NaN detection, early stopping.
"""

import logging
import math
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)


def train_linear_probe(
    features: Tensor,
    labels: Tensor,
    num_classes: int,
    config: dict,
) -> dict:
    """
    Train linear classifier on frozen features with LR grid search.

    Args:
        features: (N, D) tensor of frozen backbone features
        labels: (N,) tensor of integer labels
        num_classes: Number of output classes
        config: Must contain 'device'; optionally 'epochs' (default 100),
                'lr_grid' (default [0.001, 0.01, 0.1, 1.0])

    Returns:
        dict with keys: accuracy, best_lr, train_loss_curve
    """
    device = torch.device(config.get("device", "cpu"))
    epochs = config.get("epochs", 100)
    lr_grid = config.get("lr_grid", [0.001, 0.01, 0.1, 1.0])

    # 80/20 train/val split
    dataset = TensorDataset(features, labels)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    feature_dim = features.shape[1]
    best_accuracy = 0.0
    best_lr = lr_grid[0]
    best_loss_curve: list[float] = []

    for lr in lr_grid:
        classifier = nn.Linear(feature_dim, num_classes).to(device)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        loss_curve: list[float] = []

        for epoch in range(epochs):
            classifier.train()
            epoch_loss = 0.0
            n_batches = 0
            for feats, lbls in train_loader:
                feats, lbls = feats.to(device), lbls.to(device)
                logits = classifier(feats)
                loss = criterion(logits, lbls)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            loss_curve.append(epoch_loss / max(n_batches, 1))

        # Evaluate on validation set
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats, lbls = feats.to(device), lbls.to(device)
                preds = classifier(feats).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        accuracy = correct / max(total, 1)
        logger.info(f"  Linear probe LR={lr}: accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
            best_loss_curve = loss_curve

    return {
        "accuracy": best_accuracy,
        "best_lr": best_lr,
        "train_loss_curve": best_loss_curve,
    }


def knn_evaluate(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    k: int = 20,
) -> float:
    """
    k-NN classification using cosine similarity with majority vote.

    Args:
        train_features: (N_train, D)
        train_labels: (N_train,)
        test_features: (N_test, D)
        test_labels: (N_test,)
        k: Number of nearest neighbors

    Returns:
        Classification accuracy as float in [0, 1]
    """
    # L2-normalize features for cosine similarity
    train_normed = nn.functional.normalize(train_features, dim=1)
    test_normed = nn.functional.normalize(test_features, dim=1)

    correct = 0
    total = test_labels.size(0)

    # Process in chunks to avoid OOM on large datasets
    chunk_size = 512
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = test_normed[start:end]  # (chunk, D)

        # Cosine similarity: (chunk, N_train)
        sim = torch.mm(chunk, train_normed.t())
        _, topk_indices = sim.topk(k, dim=1)  # (chunk, k)

        topk_labels = train_labels[topk_indices]  # (chunk, k)

        # Majority vote per test sample
        for i in range(topk_labels.size(0)):
            neighbor_labels = topk_labels[i]
            predicted = neighbor_labels.mode().values.item()
            if predicted == test_labels[start + i].item():
                correct += 1

    accuracy = correct / max(total, 1)
    logger.info(f"  k-NN (k={k}): accuracy={accuracy:.4f}")
    return accuracy


def train_from_scratch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
) -> dict:
    """
    Train model from scratch with NaN detection, early stopping, and timeout.

    Uses AdamW for DeiT (with gradient clipping), SGD for ResNet.
    Includes cosine LR schedule with linear warmup.

    Args:
        model: Randomly initialized model (all params trainable)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Must contain 'device', 'arch'; optionally 'epochs' (200),
                'lr' (0.001), 'warmup_epochs' (10), 'gradient_clip' (1.0),
                'early_stopping_patience' (20), 'timeout_hours' (4)

    Returns:
        dict with: best_accuracy, train_curve, val_curve, converged
    """
    device = torch.device(config.get("device", "cpu"))
    arch = config.get("arch", "resnet34")
    epochs = config.get("epochs", 200)
    lr = config.get("lr", 0.001)
    warmup_epochs = config.get("warmup_epochs", 10)
    gradient_clip = config.get("gradient_clip", 1.0)
    patience = config.get("early_stopping_patience", 20)
    timeout_hours = config.get("timeout_hours", 4)
    max_nan_restarts = 3

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    current_lr = lr
    nan_restarts = 0
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600

    best_accuracy = 0.0
    train_curve: list[float] = []
    val_curve: list[float] = []
    epochs_without_improvement = 0
    converged = True

    while nan_restarts <= max_nan_restarts:
        # Setup optimizer: AdamW for DeiT, SGD for ResNet
        if "deit" in arch:
            optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=current_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )

        # Cosine schedule with linear warmup
        def warmup_cosine_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

        hit_nan = False
        for epoch in range(epochs):
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(f"Timeout reached ({timeout_hours}h). Stopping training.")
                converged = False
                break

            # --- Train epoch ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    hit_nan = True
                    break

                optimizer.zero_grad()
                loss.backward()

                if "deit" in arch:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if hit_nan:
                break

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_curve.append(avg_train_loss)

            # --- Validation ---
            val_acc = _validate(model, val_loader, device)
            val_curve.append(val_acc)

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % 20 == 0:
                logger.info(
                    f"  Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, "
                    f"val_acc={val_acc:.4f}, best={best_accuracy:.4f}"
                )

            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

            # Periodic MPS cleanup
            if device.type == "mps" and (epoch + 1) % 10 == 0:
                torch.mps.empty_cache()

        if hit_nan:
            nan_restarts += 1
            if nan_restarts > max_nan_restarts:
                logger.error(f"  NaN detected {nan_restarts} times. Giving up.")
                converged = False
                break
            current_lr = current_lr / 10.0
            logger.warning(
                f"  NaN detected — restart {nan_restarts}/{max_nan_restarts} with LR={current_lr}"
            )
            train_curve.clear()
            val_curve.clear()
            best_accuracy = 0.0
            epochs_without_improvement = 0

            # Re-initialize model weights
            for m in model.modules():
                reset_fn = getattr(m, "reset_parameters", None)
                if reset_fn is not None:
                    reset_fn()
            continue
        else:
            break

    return {
        "best_accuracy": best_accuracy,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "converged": converged,
    }


def _validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Run validation and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)
