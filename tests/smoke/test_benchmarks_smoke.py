"""Smoke tests for benchmark evaluation."""

import numpy as np

from gas_storage_rl.baselines.lsmc import LSMCBenchmark
from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.baselines.random_policy import RandomPolicy
from gas_storage_rl.baselines.rule_based_policy import RuleBasedPolicy
from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths


def test_benchmarks_smoke() -> None:
    """Random, rule-based, LSMC, and perfect foresight run on tiny paths."""
    paths = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]], dtype=np.float32)
    dataset = PathDataset(
        {"train": paths, "validation": paths, "test": paths},
        {"train": 1, "validation": 2, "test": 3},
    )
    params = StorageParams(capacity=2.0)
    env = GasStorageEnv(dataset, "validation", params, fixed_path_id=0)
    random_metrics, _ = evaluate_policy_on_paths(env, RandomPolicy(seed=1), path_ids=[0])
    rule = RuleBasedPolicy.from_training_prices(paths, capacity=2.0, episode_length=3)
    rule_metrics, _ = evaluate_policy_on_paths(env, rule, path_ids=[0])
    lsmc_metrics = LSMCBenchmark(params, env.lambda_terminal).fit(paths).evaluate(paths)
    pf_metrics = PerfectForesightBaseline(params, env.lambda_terminal).evaluate_paths(paths)
    assert random_metrics["split"] == "validation"
    assert rule_metrics["split"] == "validation"
    assert "mean_return_raw" in lsmc_metrics
    assert "mean_return_raw" in pf_metrics
