"""Tests for HPO search-space construction."""

from __future__ import annotations

from gas_storage_rl.hpo.search_spaces import (
    reference_trial_params,
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
    assert params["n_steps"] == 1024
    assert params["batch_size"] <= params["n_steps"]
    assert params["gamma"] == 1.0
    assert params["normalize_advantage"] is True
    assert params["policy_kwargs"]["activation_fn"] in {"tanh", "relu"}
    assert "ortho_init" in params["policy_kwargs"]
    assert "log_std_init" not in params["policy_kwargs"]


def test_sac_search_space_tunes_auto_entropy_initialization() -> None:
    """SAC HPO can tune the auto entropy-coefficient initialization."""
    params = suggest_hyperparameters(DeterministicTrial(), "sac")

    assert isinstance(params["ent_coef"], str)
    assert params["ent_coef"] == "auto"
    assert params["target_entropy"] == "auto"
    assert params["use_sde"] is False
    assert params["sde_sample_freq"] == -1
    assert params["gamma"] == 1.0
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
    assert params["action_noise"]["sigma"] == 0.1
    assert params["batch_size"] == 128
    assert params["train_freq"] == 1
    assert params["gradient_steps"] == 1
    assert params["policy_delay"] == 2
    assert params["gamma"] == 1.0


def test_td3_search_space_excludes_expensive_update_settings() -> None:
    """TD3 HPO uses bounded train/update combos and policy delay values."""
    params = suggest_hyperparameters(LastChoiceTrial(), "td3")

    assert params["batch_size"] == 256
    assert params["train_freq"] == 8
    assert params["gradient_steps"] == 2
    assert params["policy_delay"] == 2
    assert params["policy_kwargs"]["net_arch"] == [256, 256]


def test_reward_scale_multiplier_uses_shared_categorical_space() -> None:
    """Reward scaling is tuned as a shared HPO parameter."""
    multiplier = suggest_reward_scale_multiplier(DeterministicTrial())

    assert multiplier == 1.0


def test_reference_trial_params_use_sb3_defaults_with_gamma_one() -> None:
    """The enqueued reference trial represents SB3 defaults with gamma fixed."""
    ppo_params = suggest_hyperparameters(DeterministicTrial(), "ppo")
    sac_reference = reference_trial_params("sac")
    td3_reference = reference_trial_params("td3")

    assert ppo_params["gamma"] == 1.0
    assert sac_reference["buffer_size"] == 1_000_000
    assert td3_reference["buffer_size"] == 1_000_000
    assert sac_reference["reward_scale_multiplier"] == 1.0
    assert td3_reference["net_arch"] == "td3_default"


def test_search_space_description_documents_phase_one_design() -> None:
    """Search-space metadata documents the fixed design parameters."""
    description = search_space_description()

    assert description["method"] == "Optuna TPE"
    assert description["direction"] == "maximize"
    assert "reward_scale_multiplier" in description["shared_tuned_parameters"]
    assert "dataset_config" in description["fixed_design_parameters"]
    assert "gamma = 1.0" in description["fixed_design_parameters"]
    assert description["sac"]["batch_size"] == "categorical [128, 256]"
    assert "train_update_combo" in description["sac"]
    assert description["td3"]["policy_delay"] == "fixed 2"
