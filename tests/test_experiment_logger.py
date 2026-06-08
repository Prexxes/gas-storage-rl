"""Tests for persistent experiment logging."""

from __future__ import annotations

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
