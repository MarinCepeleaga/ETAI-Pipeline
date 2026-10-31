"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full metrics, classification, confusion-matrix, and fairness report to a timestamped file in `results/`, and keeps a separate CSV scoreboard for comparing runs.
"""
import os
from datetime import datetime
from typing import Optional

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Append one run to the scoreboard CSV and return all recorded runs."""
    runs = (
        pd.read_csv(path, dtype={"run": str})
        if os.path.exists(path)
        else pd.DataFrame()
    )
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(
    results_dir: str,
    config: dict,
    report_text: str,
    run_id: Optional[str] = None,
) -> str:
    """
    Write one run's full report to a timestamped file and return its path.

    The optional run ID lets the report and scoreboard share the same identifier.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")
    tuning_config = config.get("tuning", {})
    cv_config = config.get("cv", {})
    selection_metric = cv_config.get(
        "selection_metric", tuning_config.get("scoring", "accuracy")
    )

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model']['params']}\n"
        f"Test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        f"Feature engineering enabled: "
        f"{config.get('feature_engineering', {}).get('enabled', True)}\n"
        f"CV metrics: {cv_config.get('metrics', 'n/a')}\n"
        f"Selection metric: {selection_metric}\n"
        f"Tuning: enabled={tuning_config.get('enabled', False)}, "
        f"trials={tuning_config.get('n_trials', 'n/a')}, "
        f"folds={tuning_config.get('n_splits', 'n/a')}, "
        f"seed={tuning_config.get('random_state', 'n/a')}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
