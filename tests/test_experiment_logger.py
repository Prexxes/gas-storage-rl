"""Tests for persistent experiment logging."""

from __future__ import annotations

import json
import re

from gas_storage_rl.logging.experiment_logger import ExperimentLogger


def test_experiment_logger_run_id_includes_environment_algorithm_and_hash(
    tmp_path,
) -> None:
    """Run ids include readable environment and algorithm components."""
    config = {
        "environment_config": {"environment_name": "deterministic"},
        "agent_config": {"algorithm_name": "ppo"},
    }

    logger = ExperimentLogger(tmp_path, config)

    assert re.fullmatch(
        r"\d{8}-\d{6}-deterministic-ppo-[0-9a-f]{8}",
        logger.run_id,
    )
    assert logger.run_dir.name == logger.run_id


def test_experiment_logger_metadata_includes_run_id_components(tmp_path) -> None:
    """Metadata stores the readable run id components."""
    config = {
        "environment_config": {"environment_name": "seasonal env"},
        "agent_config": {"algorithm_name": "sac"},
    }

    logger = ExperimentLogger(tmp_path, config)
    metadata = logger.metadata()

    assert logger.run_id.endswith(f"{logger.config_hash}")
    assert "-seasonal_env-sac-" in logger.run_id
    assert metadata["environment_name"] == "seasonal_env"
    assert metadata["algorithm_name"] == "sac"
    assert "git_dirty" in metadata
    assert "user" in metadata


def test_experiment_logger_finalize_metadata_adds_end_time(tmp_path) -> None:
    """Finalized metadata is written with an end timestamp."""
    config = {
        "environment_config": {"environment_name": "deterministic"},
        "agent_config": {"algorithm_name": "td3"},
    }
    logger = ExperimentLogger(tmp_path, config)
    metadata = logger.metadata()

    finalized = logger.finalize_metadata(metadata)

    assert "end_time" in finalized
    with (logger.run_dir / "metadata.json").open("r", encoding="utf-8") as file:
        written = json.load(file)
    assert written["end_time"] == finalized["end_time"]
    assert written["start_time"] == metadata["start_time"]
