"""Tests for HPO search-space construction."""

from __future__ import annotations

from gas_storage_rl.hpo.search_spaces import (
    search_space_description,
    suggest_hyperparameters,
    suggest_reward_scale_multiplier,
)


class DeterministicTrial:
    """Trial stub that selects the first offered value or midpoint."""

    def suggest_categorical(self, name, choices):
        """Returns deterministic categorical values."""
        del name
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        """Returns deterministic float values."""
        del name, log
        return (float(low) + float(high)) / 2.0


class NormalNoiseTrial(DeterministicTrial):
    """Trial stub that selects TD3 normal action noise."""

    def suggest_categorical(self, name, choices):
        """Returns normal action noise when that branch is requested."""
        if name == "action_noise_mode":
            return "normal"
        return super().suggest_categorical(name, choices)


class LastChoiceTrial(DeterministicTrial):
    """Trial stub that selects the last offered categorical value."""

    def suggest_categorical(self, name, choices):
        """Returns the last categorical choice."""
        del name
        return choices[-1]


def test_ppo_search_space_returns_json_safe_policy_kwargs() -> None:
    """PPO HPO parameters use JSON-safe activation names."""
    params = suggest_hyperparameters(DeterministicTrial(), "ppo")

    assert params["policy"] == "MlpPolicy"
    assert params["n_steps"] == 256
    assert params["batch_size"] <= params["n_steps"]
    assert params["policy_kwargs"]["activation_fn"] in {"tanh", "relu"}
    assert "ortho_init" in params["policy_kwargs"]
    assert "log_std_init" in params["policy_kwargs"]


def test_sac_search_space_tunes_auto_entropy_initialization() -> None:
    """SAC HPO can tune the auto entropy-coefficient initialization."""
    params = suggest_hyperparameters(DeterministicTrial(), "sac")

    assert isinstance(params["ent_coef"], str)
    assert params["ent_coef"].startswith("auto_")
    assert "target_entropy" in params
    assert params["batch_size"] == 128
    assert params["train_freq"] == 1
    assert params["gradient_steps"] == 1


def test_sac_search_space_excludes_expensive_update_settings() -> None:
    """SAC HPO avoids high batch sizes and high gradient-step counts."""
    params = suggest_hyperparameters(LastChoiceTrial(), "sac")

    assert params["batch_size"] == 256
    assert params["train_freq"] == 8
    assert params["gradient_steps"] == 2


def test_td3_search_space_uses_json_safe_action_noise() -> None:
    """TD3 action noise is stored as config data, not a Python object."""
    params = suggest_hyperparameters(NormalNoiseTrial(), "td3")

    assert params["action_noise"]["type"] == "normal"
    assert params["action_noise"]["sigma"] > 0.0
    assert params["batch_size"] == 128
    assert params["train_freq"] == 1
    assert params["gradient_steps"] == 1
    assert params["policy_delay"] == 2


def test_td3_search_space_excludes_expensive_update_settings() -> None:
    """TD3 HPO uses bounded train/update combos and policy delay values."""
    params = suggest_hyperparameters(LastChoiceTrial(), "td3")

    assert params["batch_size"] == 256
    assert params["train_freq"] == 8
    assert params["gradient_steps"] == 2
    assert params["policy_delay"] == 3


def test_reward_scale_multiplier_uses_shared_categorical_space() -> None:
    """Reward scaling is tuned as a shared HPO parameter."""
    multiplier = suggest_reward_scale_multiplier(DeterministicTrial())

    assert multiplier == 0.25


def test_search_space_description_documents_phase_one_design() -> None:
    """Search-space metadata documents the fixed design parameters."""
    description = search_space_description()

    assert description["method"] == "Optuna TPE"
    assert description["direction"] == "maximize"
    assert "reward_scale_multiplier" in description["shared_tuned_parameters"]
    assert "dataset_config" in description["fixed_design_parameters"]
    assert description["sac"]["batch_size"] == "categorical [128, 256]"
    assert "train_update_combo" in description["sac"]
    assert description["td3"]["policy_delay"] == "categorical [2, 3]"
