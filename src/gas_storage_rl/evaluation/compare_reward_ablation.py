"""Compare paired reward-function ablation experiment groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "mean_return_raw",
    "risk_adjusted_return_raw",
    "std_return_raw",
    "mean_terminal_deviation",
    "mean_cumulative_cashflow",
    "mean_number_of_constrained_actions",
)


def compare_reward_ablation(
    old_runs_csv: str | Path,
    new_runs_csv: str | Path,
    *,
    old_label: str = "economic_terminal",
    new_label: str = "mark_to_market",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Compares two experiment-group manifests by seed index."""
    old_runs = _load_completed_runs(Path(old_runs_csv))
    new_runs = _load_completed_runs(Path(new_runs_csv))
    paired_seed_indices = sorted(set(old_runs) & set(new_runs))
    if not paired_seed_indices:
        raise ValueError("No completed seed_index pairs found")

    rows = []
    for seed_index in paired_seed_indices:
        old_summary = _load_final_summary(old_runs[seed_index]["run_dir"])
        new_summary = _load_final_summary(new_runs[seed_index]["run_dir"])
        old_validation = old_summary["validation"]
        new_validation = new_summary["validation"]
        row: dict[str, Any] = {
            "seed_index": seed_index,
            "old_reward_function": old_label,
            "new_reward_function": new_label,
            "old_run_dir": str(old_runs[seed_index]["run_dir"]),
            "new_run_dir": str(new_runs[seed_index]["run_dir"]),
            "old_agent_seed": old_summary.get("agent_seed", ""),
            "new_agent_seed": new_summary.get("agent_seed", ""),
            "old_env_seed": old_summary.get("env_seed", ""),
            "new_env_seed": new_summary.get("env_seed", ""),
        }
        for metric in metrics:
            old_value = float(old_validation[metric])
            new_value = float(new_validation[metric])
            row[f"old_{metric}"] = old_value
            row[f"new_{metric}"] = new_value
            row[f"delta_{metric}"] = new_value - old_value
        rows.append(row)

    return {
        "old_reward_function": old_label,
        "new_reward_function": new_label,
        "n_pairs": len(rows),
        "paired_seed_indices": paired_seed_indices,
        "metrics": {
            metric: _summarize_delta(rows, metric)
            for metric in metrics
        },
        "rows": rows,
    }


def write_comparison_outputs(
    comparison: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Writes paired rows and aggregate summary files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "paired_reward_ablation.csv"
    summary_path = output / "reward_ablation_summary.json"

    rows = comparison["rows"]
    with rows_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {key: value for key, value in comparison.items() if key != "rows"}
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return {"rows_csv": str(rows_path), "summary_json": str(summary_path)}


def _load_completed_runs(runs_csv: Path) -> dict[int, dict[str, Any]]:
    """Loads usable runs from an experiment-group runs.csv file."""
    runs_by_seed = {}
    with runs_csv.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("status") not in {"completed", "skipped"}:
                continue
            seed_index = int(row["seed_index"])
            run_dir = _resolve_run_dir(row["run_dir"], runs_csv)
            if not (run_dir / "final_summary.json").exists():
                continue
            runs_by_seed[seed_index] = {**row, "run_dir": run_dir}
    return runs_by_seed


def _resolve_run_dir(run_dir: str, runs_csv: Path) -> Path:
    """Resolves a run directory stored in a group manifest."""
    path = Path(run_dir)
    if path.is_absolute() or path.exists():
        return path
    candidate = runs_csv.parent / path
    return candidate if candidate.exists() else path


def _load_final_summary(run_dir: Path) -> dict[str, Any]:
    """Loads one run's final summary."""
    with (run_dir / "final_summary.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _summarize_delta(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Summarizes paired metric deltas."""
    deltas = [float(row[f"delta_{metric}"]) for row in rows]
    positives = [delta for delta in deltas if delta > 0.0]
    return {
        "mean_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "positive_fraction": len(positives) / len(deltas),
    }


def main() -> None:
    """Runs reward ablation comparison from two group manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-runs-csv", required=True)
    parser.add_argument("--new-runs-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-label", default="economic_terminal")
    parser.add_argument("--new-label", default="mark_to_market")
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    args = parser.parse_args()

    comparison = compare_reward_ablation(
        args.old_runs_csv,
        args.new_runs_csv,
        old_label=args.old_label,
        new_label=args.new_label,
        metrics=tuple(args.metrics),
    )
    outputs = write_comparison_outputs(comparison, args.output_dir)
    print(json.dumps({**outputs, "summary": comparison["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
