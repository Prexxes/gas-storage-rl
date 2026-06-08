"""Create example price-path plots for the configured synthetic environments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gas_storage_rl.prices.generators import generate_price_paths, split_seeds
from gas_storage_rl.training.config import load_config

SYNTHETIC_ENVIRONMENTS = {
    "deterministic": "Deterministic seasonal raw price paths",
    "ou": "Seasonal OU raw price paths",
    "jump": "Seasonal OU jump raw price paths",
}
SPLITS = ("pretrain", "train", "validation", "test")


def main() -> None:
    """Writes raw price-path plots for the synthetic environment in a config."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/ou_c30.yaml",
        help="Config whose synthetic price environment should be plotted.",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=50,
        help="Number of raw price paths to generate and plot.",
    )
    parser.add_argument(
        "--path-length",
        type=int,
        help=(
            "Raw path length to generate. Defaults to episode_length plus "
            "max_start_offset from the config."
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=SPLITS,
        help="Dataset split seed to use. Test paths are plotted only with --split test.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base dataset seed.")
    parser.add_argument(
        "--output-dir",
        default="reports/price_path_plots",
        help="Directory for generated PNG files.",
    )
    args = parser.parse_args()
    if args.n_paths <= 0:
        raise ValueError("n_paths must be positive")

    config = load_config(args.config)
    env_config = config["environment_config"]
    dataset_config = config["dataset_config"]
    environment_name = env_config["environment_name"]
    if environment_name not in SYNTHETIC_ENVIRONMENTS:
        raise ValueError(f"Unsupported synthetic environment_name: {environment_name}")
    path_length = _raw_path_length(env_config, dataset_config, args.path_length)
    if path_length <= 0:
        raise ValueError("path_length must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = generate_price_paths(
        environment_name=environment_name,
        n_paths=args.n_paths,
        episode_length=path_length,
        seed=_seed_for_split(args.seed, args.split),
        params=config["price_process_config"],
    )
    _plot_paths(
        paths,
        f"{SYNTHETIC_ENVIRONMENTS[environment_name]} ({args.split})",
        out_dir / (
            f"{environment_name}_{args.split}_raw_{path_length}_price_paths.png"
        ),
    )
    print(out_dir.resolve())


def _raw_path_length(
    env_config: dict,
    dataset_config: dict,
    path_length: int | None,
) -> int:
    """Returns the raw path length requested by the plot CLI."""
    if path_length is not None:
        return int(path_length)
    episode_length = int(env_config["episode_length"])
    max_start_offset = int(dataset_config.get("max_start_offset", episode_length - 1))
    return episode_length + max_start_offset


def _seed_for_split(dataset_seed: int, split: str) -> int:
    """Returns the deterministic dataset seed for a split."""
    return split_seeds(dataset_seed, include_pretrain=True)[split]


def _plot_paths(paths: np.ndarray, title: str, output_path: Path) -> None:
    """Writes one raw price-path plot."""
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=140)
    axis.plot(paths.T, color="#2f6f9f", alpha=0.16, linewidth=0.9)
    axis.plot(
        np.mean(paths, axis=0),
        color="#111111",
        linewidth=2.2,
        label="Mean",
    )
    if paths.shape[1] > 365:
        axis.axvline(
            365,
            color="#8a3ffc",
            linestyle="--",
            linewidth=1.4,
            label="New year",
        )
    axis.set_title(title)
    axis.set_xlabel("Raw simulation day")
    axis.set_ylabel("Gas spot price")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
