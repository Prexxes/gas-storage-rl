"""Create comparison plots across RL runs and benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    """Creates learning-curve, return, and regret comparison plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-run-dir", action="append", default=[])
    parser.add_argument("--benchmark-run-dir")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rl_run_dir or args.benchmark_run_dir:
        learning = _load_learning_curves(args.rl_run_dir, args.benchmark_run_dir)
        if not learning.empty:
            _save(
                plot_learning_curves(learning, args.split),
                output_dir / f"learning_curves_{args.split}.png",
            )

    final = _load_final_episode_metrics(
        args.rl_run_dir,
        args.benchmark_run_dir,
        args.split,
    )
    if not final.empty:
        _save(
            plot_final_return_violins(final),
            output_dir / f"final_return_violin_{args.split}.png",
        )
        regret = relative_regret_to_perfect_foresight(final)
        if not regret.empty:
            _save(
                plot_regret_violins(regret),
                output_dir / f"relative_regret_violin_{args.split}.png",
            )
    print(f"Saved comparison plots to {output_dir.resolve()}")


def plot_learning_curves(metrics: pd.DataFrame, split: str) -> plt.Figure:
    """Plots one line per RL run and benchmark reference lines."""
    data = metrics[metrics["split"] == split].copy()
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, group in data.groupby("series_label", sort=False):
        group = group.sort_values("total_training_env_steps")
        linestyle = "--" if bool(group["is_benchmark"].iloc[0]) else "-"
        alpha = 0.85 if bool(group["is_benchmark"].iloc[0]) else 1.0
        axis.plot(
            group["total_training_env_steps"],
            group["mean_return_raw"],
            label=label,
            linestyle=linestyle,
            alpha=alpha,
        )
    axis.set_xlabel("Total environment steps")
    axis.set_ylabel("Mean raw return")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    return figure


def plot_final_return_violins(metrics: pd.DataFrame) -> plt.Figure:
    """Plots final episode-return distributions by method."""
    return _violin_plot(
        metrics,
        value_column="episode_return_raw",
        ylabel="Episode raw return",
    )


def plot_regret_violins(regret: pd.DataFrame) -> plt.Figure:
    """Plots relative regret distributions against perfect foresight."""
    return _violin_plot(
        regret,
        value_column="relative_regret_to_pf",
        ylabel="Relative regret to perfect foresight",
    )


def relative_regret_to_perfect_foresight(
    metrics: pd.DataFrame,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Computes per-episode relative regret against perfect foresight."""
    key = ["split", "path_id"]
    pf = metrics[metrics["method"] == "perfect_foresight"][
        key + ["episode_return_raw"]
    ].rename(columns={"episode_return_raw": "perfect_foresight_return"})
    compared = metrics[metrics["method"] != "perfect_foresight"].merge(pf, on=key)
    denominator = compared["perfect_foresight_return"].abs().clip(lower=epsilon)
    compared["relative_regret_to_pf"] = (
        compared["perfect_foresight_return"] - compared["episode_return_raw"]
    ) / denominator
    return compared


def _violin_plot(
    metrics: pd.DataFrame,
    value_column: str,
    ylabel: str,
) -> plt.Figure:
    """Creates a Matplotlib violin plot grouped by method."""
    methods = list(dict.fromkeys(metrics["method"].astype(str)))
    values = [
        metrics.loc[metrics["method"] == method, value_column].dropna().to_numpy()
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(max(8, len(methods) * 0.9), 5))
    axis.violinplot(values, showmeans=True, showmedians=True)
    axis.set_xticks(np.arange(1, len(methods) + 1), methods, rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _load_learning_curves(
    rl_run_dirs: list[str],
    benchmark_run_dir: str | None,
) -> pd.DataFrame:
    """Loads RL evaluations and benchmark evaluation references."""
    frames = []
    for run_dir_text in rl_run_dirs:
        run_dir = Path(run_dir_text)
        path = run_dir / "evaluations.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        method = str(frame.get("algorithm_name", pd.Series([run_dir.name])).iloc[0])
        frame["method"] = method
        frame["series_label"] = f"{method}:{run_dir.name}"
        frame["run_dir"] = str(run_dir)
        frame["is_benchmark"] = False
        frames.append(frame)
    if benchmark_run_dir:
        path = Path(benchmark_run_dir) / "benchmark_evaluations.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["series_label"] = frame["method"].astype(str)
            frame["run_dir"] = str(Path(benchmark_run_dir))
            frame["is_benchmark"] = True
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_final_episode_metrics(
    rl_run_dirs: list[str],
    benchmark_run_dir: str | None,
    split: str,
) -> pd.DataFrame:
    """Loads final episode metrics from RL and benchmark runs."""
    frames = []
    for run_dir_text in rl_run_dirs:
        run_dir = Path(run_dir_text)
        path = run_dir / f"final_episode_metrics_{split}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["run_dir"] = str(run_dir)
            frames.append(frame)
    if benchmark_run_dir:
        path = Path(benchmark_run_dir) / f"final_episode_metrics_{split}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["run_dir"] = str(Path(benchmark_run_dir))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _default_output_dir(args: Any) -> Path:
    """Returns a default comparison plot directory."""
    if args.benchmark_run_dir:
        return Path(args.benchmark_run_dir) / "plots" / "comparison"
    if args.rl_run_dir:
        return Path(args.rl_run_dir[0]) / "plots" / "comparison"
    return Path("plots") / "comparison"


def _save(figure: plt.Figure, path: Path) -> None:
    """Saves and closes a Matplotlib figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
