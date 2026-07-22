"""Run experiment groups with HPO-selected settings on a target config."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from gas_storage_rl.training.config import load_config
from gas_storage_rl.training.run_experiment_group import run_experiment_group


def transfer_best_agent_settings(
    best_config: dict[str, Any],
    target_config: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    """Copies HPO-selected agent settings and reward scale to a target config.

    Args:
        best_config: HPO ``best_config.json`` contents.
        target_config: Base experiment configuration to run.
        algorithm: Algorithm whose hyperparameters should be transferred.

    Returns:
        Target config copy with transferred settings.

    Raises:
        ValueError: If the requested algorithm or reward scale is unavailable.

    """
    if algorithm not in best_config.get("agent_config", {}):
        raise ValueError(
            f"best_config does not contain agent_config for algorithm: {algorithm}"
        )
    if "reward_scale" not in best_config.get("environment_config", {}):
        raise ValueError("best_config does not contain environment_config.reward_scale")

    output = copy.deepcopy(target_config)
    output.setdefault("agent_config", {})[algorithm] = copy.deepcopy(
        best_config["agent_config"][algorithm]
    )
    output["agent_config"]["algorithm_name"] = algorithm
    output.setdefault("environment_config", {})["reward_scale"] = float(
        best_config["environment_config"]["reward_scale"]
    )
    return output


def run_transferred_best_config_group(
    best_config: dict[str, Any],
    target_config: dict[str, Any],
    target_config_name: str,
    algorithm: str,
    *,
    n_seeds: int | None = None,
    seed_indices: list[int] | None = None,
    pretrained_policy: str | Path | None = None,
    rerun: bool = False,
) -> dict[str, Any]:
    """Runs an experiment group after transferring HPO-selected settings.

    Args:
        best_config: HPO ``best_config.json`` contents.
        target_config: Base experiment configuration to run.
        target_config_name: Human-readable target configuration name.
        algorithm: Algorithm whose hyperparameters should be transferred.
        n_seeds: Number of seed repetitions.
        seed_indices: Seed repetition indices to run.
        pretrained_policy: Pretrained policy value.
        rerun: Rerun value.

    Returns:
        Experiment group summary.

    """
    transferred_config = transfer_best_agent_settings(
        best_config,
        target_config,
        algorithm,
    )
    return run_experiment_group(
        transferred_config,
        f"{target_config_name}-hpo-transfer",
        algorithm,
        n_seeds=n_seeds,
        seed_indices=seed_indices,
        pretrained_policy=pretrained_policy,
        rerun=rerun,
    )


def _load_best_config(path: str | Path) -> dict[str, Any]:
    """Loads an HPO best config JSON file.

    Args:
        path: Filesystem path to ``best_config.json``.

    Returns:
        Parsed JSON configuration dictionary.

    """
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Writes a YAML document.

    Args:
        path: Filesystem path to write.
        payload: Serializable payload to persist.

    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def main() -> None:
    """Runs a target config with settings transferred from HPO best_config.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best-config",
        required=True,
        help="Reference runs/hpo/<study_id>/best_config.json.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Target config YAML, for example configs/deterministic_c30.yaml.",
    )
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--seed-indices", type=int, nargs="+")
    parser.add_argument("--pretrained-policy")
    parser.add_argument(
        "--output-config",
        help="Optional path to write the merged config before running.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run seeds even if completed runs with the same effective configs exist.",
    )
    args = parser.parse_args()

    best_config = _load_best_config(args.best_config)
    config_path = Path(args.config)
    target_config = load_config(config_path)
    transferred_config = transfer_best_agent_settings(
        best_config,
        target_config,
        args.algorithm,
    )
    if args.output_config:
        _write_yaml(args.output_config, transferred_config)

    summary = run_experiment_group(
        transferred_config,
        f"{config_path.stem}-hpo-transfer",
        args.algorithm,
        n_seeds=args.n_seeds,
        seed_indices=args.seed_indices,
        pretrained_policy=args.pretrained_policy,
        rerun=args.rerun,
    )
    print(json.dumps(summary, indent=2))
    if summary["failed_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
