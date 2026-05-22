#!/usr/bin/env python3
"""Run the three-score embedding CSVs through AutoGluon on the Mini cluster."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = Path("/Users/johnennis/aigora/dev/tabpfn-launcher")
FEATURES = ["visual_similarity", "texture_similarity", "flavor_similarity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-dir", default=DEFAULT_LAUNCHER, type=Path)
    parser.add_argument("--manifest", default=ROOT / "data" / "embedding_baselines" / "tabpfn_inputs" / "manifest_seed42.json", type=Path)
    parser.add_argument("--outputs-dir", default=ROOT / "data" / "embedding_baselines" / "autogluon_outputs", type=Path)
    parser.add_argument("--summary-csv", default=ROOT / "data" / "embedding_baselines" / "three_score_embedding_autogluon_summary.csv", type=Path)
    parser.add_argument("--node", default="v2")
    parser.add_argument("--time-limit", default=90, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--keep-model-dir", action="store_true", help="Keep copied AutoGluon model directories.")
    return parser.parse_args()


def short_feature_name(feature_set: str) -> str:
    if feature_set.startswith("three_modality_scores_"):
        feature_set = feature_set.removeprefix("three_modality_scores_")
    return feature_set.replace("_", "-")


def absolute_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def metric_abs(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else abs(float(value))


def run_one(args: argparse.Namespace, feature_set: str, csv_path: Path) -> dict:
    launcher = args.launcher_dir / "scripts" / "autogluon_tabular_cluster.py"
    run_id = f"embedding-{short_feature_name(feature_set)}-seed{args.random_state}"
    command = [
        sys.executable,
        str(launcher),
        "--node",
        args.node,
        "--csv",
        str(csv_path),
        "--target",
        "liking",
        "--task",
        "regression",
        "--features-json",
        json.dumps(FEATURES, separators=(",", ":")),
        "--split-column",
        "split",
        "--train-split-value",
        "train",
        "--test-split-value",
        "holdout",
        "--outputs-dir",
        str(args.outputs_dir),
        "--run-id",
        run_id,
        "--time-limit",
        str(args.time_limit),
        "--random-state",
        str(args.random_state),
        "--json",
    ]
    subprocess.run(command, cwd=args.launcher_dir, check=True)
    summary_path = args.outputs_dir / f"{run_id}-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = summary["results"][0]
    artifact_dir = Path(result["artifact_dir"])
    if not args.keep_model_dir:
        shutil.rmtree(artifact_dir / "model", ignore_errors=True)
    metrics = result["metrics"]
    leaderboard = result.get("leaderboard") or []
    return {
        "engine": "autogluon",
        "validation": "single explicit GroupShuffleSplit seed42 by consumer",
        "feature_set": short_feature_name(feature_set),
        "model": leaderboard[0].get("model") if leaderboard else "",
        "r2_mean": metrics.get("r2"),
        "r2_std": "",
        "mae_mean": metric_abs(metrics, "mean_absolute_error"),
        "rmse_mean": metric_abs(metrics, "root_mean_squared_error"),
        "pearsonr": metrics.get("pearsonr"),
        "train_rows": result.get("train_rows"),
        "test_rows": result.get("test_rows"),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for item in manifest["files"]:
        feature_set = item["feature_set"]
        csv_path = absolute_path(item["path"])
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        rows.append(run_one(args, feature_set, csv_path))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("r2_mean", ascending=False).to_csv(args.summary_csv, index=False)
    print(args.summary_csv)


if __name__ == "__main__":
    main()
