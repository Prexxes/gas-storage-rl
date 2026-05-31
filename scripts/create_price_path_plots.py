"""Create example price-path plots for the configured synthetic environments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gas_storage_rl.prices.generators import generate_price_paths


def main() -> None:
    """Writes one price-path plot for each baseline price environment."""
    out_dir = Path("reports/price_path_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [
        ("deterministic", "Deterministic seasonal price paths"),
        ("ou", "Seasonal OU price paths"),
        ("jump", "Seasonal OU jump price paths"),
    ]
    for env_name, title in settings:
        paths = generate_price_paths(
            environment_name=env_name,
            n_paths=80,
            episode_length=365,
            seed=123,
        )
        figure, axis = plt.subplots(figsize=(10, 5.5), dpi=140)
        selected = paths[:50]
        axis.plot(selected.T, color="#2f6f9f", alpha=0.16, linewidth=0.9)
        axis.plot(
            np.mean(selected, axis=0),
            color="#111111",
            linewidth=2.2,
            label="Mean",
        )
        axis.set_title(title)
        axis.set_xlabel("Day")
        axis.set_ylabel("Gas spot price")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(out_dir / f"{env_name}_price_paths.png")
        plt.close(figure)
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
