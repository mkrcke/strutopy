"""Record baseline metrics for regression testing.

Usage:
    uv run python tests/record_baseline.py
"""

import json
import os
import sys

# Allow importing from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from test_integration import _run_toy_pipeline

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "baseline_metrics.json")


def main():
    print("Running toy pipeline...")
    metrics = _run_toy_pipeline()

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline metrics written to {FIXTURE_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
