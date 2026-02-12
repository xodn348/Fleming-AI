#!/usr/bin/env python3
"""
2x2 factorial experiment runner: pre-training x architecture interaction.

Usage:
    python experiment/scripts/02_run_experiments.py --dry-run
    python experiment/scripts/02_run_experiments.py
    python experiment/scripts/02_run_experiments.py --resume
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from tqdm import tqdm

from experiment.src.evaluate import run_single_experiment, save_result
from experiment.src.utils import check_disk_space, get_device, setup_logging

logger = logging.getLogger(__name__)

ARCHITECTURES = ["deit_small", "resnet34"]
PRETRAINED = [True, False]
DATASETS = ["cifar10", "cifar100", "stl10", "flowers102", "oxford_pets"]
EVAL_METHODS = ["linear_probe", "knn"]
SEEDS = [42, 123, 456]

# Total: 2 x 2 x 5 x 2 x 3 = 120 experiments


def generate_configs(device: str) -> list[dict]:
    configs = []
    for arch in ARCHITECTURES:
        for pretrained in PRETRAINED:
            for dataset in DATASETS:
                for eval_method in EVAL_METHODS:
                    for seed in SEEDS:
                        configs.append(
                            {
                                "arch": arch,
                                "pretrained": pretrained,
                                "dataset": dataset,
                                "eval_method": eval_method,
                                "seed": seed,
                                "device": device,
                            }
                        )
    return configs


def load_completed_keys(results_dir: Path) -> set[str]:
    """Load keys of already-completed experiments for --resume."""
    results_file = results_dir / "all_results.jsonl"
    completed = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = _config_key(r)
                    completed.add(key)
                except json.JSONDecodeError:
                    continue
    return completed


def _config_key(config: dict) -> str:
    return (
        f"{config['arch']}_{config['pretrained']}_{config['dataset']}_"
        f"{config['eval_method']}_{config['seed']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run 2x2 factorial experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show configs without running")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed experiments")
    args = parser.parse_args()

    results_dir = Path.home() / "Fleming-AI" / "experiment" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(results_dir / "experiment.log")

    device = get_device()
    configs = generate_configs(str(device))

    logger.info(f"Generated {len(configs)} experiment configurations")

    if args.dry_run:
        print(f"{len(configs)} experiments planned")
        for i, cfg in enumerate(configs[:5]):
            print(f"  [{i + 1}] {cfg}")
        if len(configs) > 5:
            print(f"  ... ({len(configs) - 5} more)")
        return

    check_disk_space(min_gb=5.0)

    completed_keys: set[str] = set()
    if args.resume:
        completed_keys = load_completed_keys(results_dir)
        logger.info(f"Resuming: {len(completed_keys)} experiments already completed")

    skipped = 0
    failed = 0
    succeeded = 0

    for i, config in enumerate(tqdm(configs, desc="Experiments")):
        if args.resume and _config_key(config) in completed_keys:
            skipped += 1
            continue

        logger.info(f"[{i + 1}/{len(configs)}] Running: {config}")

        try:
            result = run_single_experiment(config)
            save_result(result, results_dir)
            succeeded += 1
            logger.info(f"  -> Accuracy: {result['accuracy']:.2%}")

            if device.type == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()

        except Exception as e:
            failed += 1
            logger.error(f"  -> FAILED: {e}", exc_info=True)
            continue

    logger.info(
        f"Complete! succeeded={succeeded}, failed={failed}, skipped={skipped}, total={len(configs)}"
    )


if __name__ == "__main__":
    main()
