"""Tests for price-path plotting CLIs."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def _load_script(path: str):
    """Loads a script module by path."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_raw_path_length_uses_episode_plus_max_start_offset() -> None:
    """Synthetic plot CLI defaults to the dataset raw path length."""
    module = _load_script("scripts/create_price_path_plots.py")

    raw_path_length = module._raw_path_length(
        {"episode_length": 365},
        {},
        None,
    )

    assert raw_path_length == 729


def test_synthetic_seed_for_split_uses_dataset_split_offsets() -> None:
    """Synthetic plot CLI uses deterministic dataset split seed offsets."""
    module = _load_script("scripts/create_price_path_plots.py")

    assert module._seed_for_split(123, "train") == 123
    assert module._seed_for_split(123, "validation") == 1_000_126
    assert module._seed_for_split(123, "test") == 2_000_126


def test_create_price_path_plots_cli_writes_requested_environment(tmp_path) -> None:
    """Synthetic plot CLI writes the selected raw price path plot."""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "scripts/create_price_path_plots.py",
            "--config",
            "configs/debug.yaml",
            "--n-paths",
            "2",
            "--path-length",
            "9",
            "--split",
            "validation",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        env=env,
    )

    assert (tmp_path / "deterministic_validation_raw_9_price_paths.png").exists()
