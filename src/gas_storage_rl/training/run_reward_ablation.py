"""Run paired old-versus-new reward function ablation groups."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from gas_storage_rl.evaluation.compare_reward_ablation import (
    compare_reward_ablation,
    write_comparison_outputs,
)
from gas_storage_rl.training.config import load_config
from gas_storage_rl.training.run_experiment_group import run_experiment_group


def run_reward_ablation(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    *,
    n_seeds: int | None = None,
    old_reward_function: str = "economic_terminal",
    new_reward_function: str = "mark_to_market",
    rerun: bool = False,
) -> dict[str, Any]:
    """Runs old and new reward groups, then writes paired seed comparison files."""
    old_config = _config_with_reward_function(config, old_reward_function)
    new_config = _config_with_reward_function(config, new_reward_function)

    old_group = run_experiment_group(
        old_config,
        f"{config_name}-{old_reward_function}",
        algorithm,
        n_seeds=n_seeds,
        rerun=rerun,
    )
    new_group = run_experiment_group(
        new_config,
        f"{config_name}-{new_reward_function}",
        algorithm,
        n_seeds=n_seeds,
        rerun=rerun,
    )

    output_dir = (
        Path(config.get("logging_config", {}).get("run_dir", "runs"))
        / "reward_ablations"
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{config_name}-{algorithm}"
    )
    comparison = compare_reward_ablation(
        Path(old_group["group_dir"]) / "runs.csv",
        Path(new_group["group_dir"]) / "runs.csv",
        old_label=old_reward_function,
        new_label=new_reward_function,
    )
    outputs = write_comparison_outputs(comparison, output_dir)
    return {
        "algorithm_name": algorithm,
        "old_group_dir": old_group["group_dir"],
        "new_group_dir": new_group["group_dir"],
        "comparison_dir": str(output_dir),
        **outputs,
        "comparison": {
            key: value for key, value in comparison.items() if key != "rows"
        },
    }


def _config_with_reward_function(
    config: dict[str, Any],
    reward_function: str,
) -> dict[str, Any]:
    """Returns a config copy with the selected reward function."""
    updated = copy.deepcopy(config)
    updated.setdefault("environment_config", {})
    updated["environment_config"]["reward_function"] = reward_function
    return updated


def main() -> None:
    """Runs a paired reward-function ablation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--old-reward-function", default="economic_terminal")
    parser.add_argument("--new-reward-function", default="mark_to_market")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run seeds even if completed runs with the same effective configs exist.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    summary = run_reward_ablation(
        load_config(config_path),
        config_path.stem,
        args.algorithm,
        n_seeds=args.n_seeds,
        old_reward_function=args.old_reward_function,
        new_reward_function=args.new_reward_function,
        rerun=args.rerun,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
