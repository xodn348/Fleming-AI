#!/usr/bin/env python3
"""
Smoke test for experiment pipeline.
Tests 3 quick configurations to validate everything works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment.src.evaluate import run_single_experiment
from experiment.src.utils import set_seed, get_device, setup_logging
import json
import time
import os

os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    device = get_device()
    results_dir = Path.home() / "Fleming-AI" / "experiment" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(results_dir / "smoke_test.log")

    print("=" * 60)
    print("SMOKE TEST - Validating Experiment Pipeline")
    print("=" * 60)

    configs = [
        {
            "name": "k-NN (pretrained)",
            "arch": "deit_small",
            "pretrained": True,
            "dataset": "cifar10",
            "eval_method": "knn",
            "seed": 42,
            "device": str(device),
        },
        {
            "name": "k-NN (from-scratch)",
            "arch": "deit_small",
            "pretrained": False,
            "dataset": "cifar10",
            "eval_method": "knn",
            "seed": 42,
            "device": str(device),
        },
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/3] Running: {config['name']}")
        print(
            f"  Config: {config['arch']}, pretrained={config['pretrained']}, {config['eval_method']}"
        )

        start = time.time()
        try:
            result = run_single_experiment(config)
            elapsed = time.time() - start

            print(f"  ✅ SUCCESS in {elapsed:.1f}s")
            print(f"  Accuracy: {result.get('accuracy', 0):.2%}")

            results.append(
                {
                    "test": config["name"],
                    "config": config,
                    "result": result,
                    "elapsed_seconds": elapsed,
                    "status": "success",
                }
            )

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ FAILED in {elapsed:.1f}s: {e}")
            results.append(
                {
                    "test": config["name"],
                    "config": config,
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                    "status": "failed",
                }
            )

    # Save results
    output_file = results_dir / "smoke_test.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"Passed: {passed}/3")
    print(f"Failed: {failed}/3")

    if failed == 0:
        print("\n✅ SMOKE TEST PASSED - Pipeline is ready for full experiments")
        print(f"Results saved to: {output_file}")
        return 0
    else:
        print("\n❌ SMOKE TEST FAILED - Fix errors before running full experiments")
        print(f"Results saved to: {output_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
