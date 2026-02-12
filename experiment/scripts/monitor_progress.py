#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter

results_file = Path.home() / "Fleming-AI" / "experiment" / "results" / "all_results.jsonl"

if not results_file.exists():
    print("No results yet")
    exit(0)

results = []
with open(results_file) as f:
    for line in f:
        results.append(json.loads(line))

print(f"=" * 60)
print(f"EXPERIMENT PROGRESS")
print(f"=" * 60)
print(f"Completed: {len(results)}/120 ({len(results) / 120 * 100:.1f}%)")

by_eval = Counter(r["eval_method"] for r in results)
print(f"\nBy evaluation method:")
for method, count in by_eval.items():
    print(f"  {method}: {count}")

by_arch = Counter(r["arch"] for r in results)
print(f"\nBy architecture:")
for arch, count in by_arch.items():
    print(f"  {arch}: {count}")

by_dataset = Counter(r["dataset"] for r in results)
print(f"\nBy dataset:")
for dataset, count in by_dataset.items():
    print(f"  {dataset}: {count}")

accs = [r["accuracy"] for r in results]
print(f"\nAccuracy range: {min(accs):.2%} - {max(accs):.2%}")
print(f"Mean accuracy: {sum(accs) / len(accs):.2%}")

latest = results[-1]
print(
    f"\nLatest: {latest['arch']}, pretrained={latest['pretrained']}, {latest['dataset']}, {latest['eval_method']}"
)
print(f"  Accuracy: {latest['accuracy']:.2%}")

if len(results) >= 2:
    first_time = results[0].get("timestamp", 0)
    last_time = results[-1].get("timestamp", 0)
    if first_time and last_time:
        elapsed = last_time - first_time
        rate = len(results) / elapsed
        remaining = 120 - len(results)
        eta_seconds = remaining / rate
        eta_hours = eta_seconds / 3600
        print(f"\nEstimated time remaining: {eta_hours:.1f} hours")
