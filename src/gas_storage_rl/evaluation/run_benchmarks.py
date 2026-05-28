"""Command-line runner for benchmark policies."""

from __future__ import annotations

import argparse
import json

from gas_storage_rl.baselines.lsmc import LSMCBenchmark
from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.baselines.random_policy import RandomPolicy
from gas_storage_rl.baselines.rule_based_policy import RuleBasedPolicy
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.training.config import build_environment, load_config


def main() -> None:
    """Runs random, rule-based, LSMC, and perfect-foresight benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset, storage_params, env_kwargs = build_environment(config)
    env = GasStorageEnv(dataset, "validation", storage_params, **env_kwargs)
    random_policy = RandomPolicy(seed=config["seeds"]["eval_seed"])
    rule_policy = RuleBasedPolicy.from_training_prices(
        dataset.get_paths("train"),
        capacity=storage_params.capacity,
        episode_length=dataset.episode_length,
    )
    from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths

    random_metrics, _ = evaluate_policy_on_paths(env, random_policy)
    rule_metrics, _ = evaluate_policy_on_paths(env, rule_policy)
    lsmc = LSMCBenchmark(storage_params, env.lambda_terminal).fit(dataset.get_paths("train"))
    pf = PerfectForesightBaseline(storage_params, env.lambda_terminal)
    output = {
        "random": random_metrics,
        "rule_based": rule_metrics,
        "lsmc": lsmc.evaluate(dataset.get_paths("validation")),
        "perfect_foresight": pf.evaluate_paths(dataset.get_paths("validation")),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
