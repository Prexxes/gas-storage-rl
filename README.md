# Gas Storage RL

This repository evaluates PPO, SAC, and TD3 for natural gas storage asset valuation on synthetic spot-price environments with increasing complexity. It is designed for a bachelor thesis comparison of performance, stability across seeds, and sample efficiency.

## Environments

The MVP uses three Gymnasium-compatible settings:

- Deterministic seasonal: `log_price_t = seasonal_log_price_t`
- Seasonal OU: `log_price_t = seasonal_log_price_t + ou_residual_t`
- Seasonal OU jump/stress: `log_price_t = seasonal_log_price_t + ou_residual_t + jump_component_t`

Prices are generated as `price_t = exp(log_price_t)` and are strictly positive. Train, validation, and test paths are generated from deterministic, disjoint seeds. Deterministic environments repeat the same seasonal path through the same dataset interface.

Generated price datasets are cached by default under `data/cache/{dataset_hash}/` as `train.npy`, `validation.npy`, `test.npy`, and `metadata.json`. The hash depends on the environment name, path counts, episode length, dataset seed, and price-process parameters. Set `dataset_config.force_regenerate: true` to overwrite an existing matching cache.

## Historical Calibration Pipeline

The repository can also calibrate synthetic prices from prepared historical CSVs while keeping calibration and backtesting chronologically separate. Monthly calibration data up to and including `2024-12-31` is used to estimate a log-seasonal calendar-month component. The twelve monthly seasonal values are evaluated as a smooth periodic Fourier curve for daily paths, avoiding hard jumps at month boundaries. Daily calibration data up to the same cutoff is deseasonalized into log residuals, then used to fit an AR(1)/OU residual process and a simple jump component from large AR(1) innovations.

Held-out historical backtesting starts at `2025-01-01`. Those backtest prices are never used for seasonality, residual, OU, or jump calibration. They are only converted into separate rolling-window episodes.

Build calibrated synthetic train/validation/test paths and historical backtest windows:

```bash
PYTHONPATH=src python -m gas_storage_rl.data.build_historical_datasets --config configs/historical_debug.yaml
```

The synthetic splits are stored under `data/cache/{dataset_hash}/`. Historical backtest windows are stored separately under `data/cache/backtest/{dataset_hash}/` as `backtest.npy` plus metadata with each episode start and end date.

Historically calibrated synthetic paths support three environment variants:

- `historical_deterministic`: calibrated monthly log seasonality only
- `historical_ou`: calibrated seasonality plus AR(1)/OU residual noise
- `historical_jump`: calibrated seasonality plus AR(1)/OU residual noise and jumps

## Storage Dynamics

The storage level satisfies `0 <= v_t <= C`. The continuous action space is `Box(-1, 1, shape=(1,))`, where positive actions inject gas and negative actions withdraw gas. Executed actions are clipped by injection and withdrawal rates and by storage boundaries. The MVP uses efficiency `1`, transaction costs `0`, leakage `0`, and no volume-dependent rates.

## Observation Space

Observations are `np.float32` vectors:

```text
[storage_level / capacity, price / price_scale, current_step / (episode_length - 1)]
```

The MVP intentionally does not include M+1 futures features.

## Reward

The raw economic cashflow is computed from the executed action:

```text
raw_cashflow_t = -executed_action_t * price_t
```

Buying gas is negative cashflow and selling gas is positive cashflow. Training uses
a mark-to-market shaped reward so injected gas can receive a learning signal before
it is sold. For non-terminal steps:

```text
mark_to_market_reward_t = v_(t+1) * (p_(t+1) - p_t)
max_withdrawable_remaining = remaining_steps * withdrawal_rate
excess_inventory = max(
    0,
    v_(t+1) - target_inventory - max_withdrawable_remaining,
)
feasibility_penalty = -lambda_feasibility * excess_inventory
shaped_reward_t = mark_to_market_reward_t + feasibility_penalty
```

At the final step a terminal inventory penalty is applied:

```text
terminal_penalty = -penalty_factor * mean_training_price * abs(v_T - target_inventory)
shaped_reward_T = raw_cashflow_T + terminal_penalty - v_T * p_T
scaled_reward = shaped_reward / reward_scale
```

The environment returns scaled shaped rewards to RL algorithms. Economic evaluation
continues to use true cashflows plus terminal penalty through `economic_reward_raw`
and `raw_cashflow` in `info`.

## Benchmarks

Implemented benchmarks are:

- `RandomPolicy`: samples uniformly from `[-1, 1]`.
- `RuleBasedPolicy`: buys below the 30 percent training quantile, sells above the 70 percent quantile, and enforces safe liquidation near maturity.
- `LSMCBenchmark`: Least-Squares Monte Carlo with action grid `[-1, 0, 1]` and polynomial continuation features.
- `PerfectForesightBaseline`: deterministic upper bound solved with `scipy.optimize.linprog`.

## Algorithms

PPO, SAC, and TD3 are created through Stable-Baselines3. RL Baselines3 Zoo may be used as a reference for implementation details and hyperparameters, but the core framework is this repository.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Run a debug experiment:

```bash
PYTHONPATH=src python -m gas_storage_rl.training.run_experiment --config configs/debug.yaml --algorithm ppo
```

Each run writes `config.json`, `metadata.json`, `metrics.csv`, `evaluations.csv`, `final_summary.json`, `final_model.zip`, and SB3 internal logs under `sb3_logs/`. The training callback logs completed training episodes to `metrics.csv`, runs periodic validation according to `training_config.eval_freq`, and saves `best_validation_model.zip`. The training command does not evaluate the test split or historical backtest split; those holdout evaluations are run manually after model selection.

Run manual holdout evaluation on the synthetic test split:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_holdout_evaluation --run-dir runs/<run_id> --split test
```

Run manual holdout evaluation on historical backtest windows:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_holdout_evaluation --run-dir runs/<run_id> --split backtest
```

Run benchmarks:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks --config configs/debug.yaml
```

Plotting helpers live in `gas_storage_rl.plotting` and return Matplotlib figures for price paths, action paths, storage levels, cumulative cashflows, price-action scatter, learning curves, and return distributions.

Create plots for a saved run:

```bash
PYTHONPATH=src python -m gas_storage_rl.plotting.plot_run --run-dir runs/<run_id> --split validation --path-id 0
```
