# Gas Storage RL

This repository evaluates PPO, SAC, and TD3 for natural gas storage asset valuation on synthetic spot-price environments with increasing complexity. It is designed for a bachelor thesis comparison of performance, stability across seeds, and sample efficiency.

## Environments

The MVP uses three Gymnasium-compatible synthetic settings:

- Deterministic seasonal: `price_t = 2 + sin(2 * pi * t / 365)`
- Seasonal OU: `price_t = 2 + sin(2 * pi * t / 365) + X_t`
- Seasonal OU jump/stress: `price_t = 2 + sin(2 * pi * t / 365) + X_t + J_t`

`X_t` is an exactly discretized additive Ornstein-Uhlenbeck process with speed
of mean reversion `1.0`, long-term mean `0.0`, volatility `1.2`, initial value
`0.0`, and a daily time step of `1 / 365`. The synthetic processes deliberately permit negative prices as a
controlled RL benchmark. Pretrain, train, validation, and test paths are
generated from deterministic, disjoint seeds. Deterministic environments repeat
the same seasonal path through the same dataset interface.

Generated price datasets are cached by default under `data/cache/{dataset_hash}/` as `pretrain.npy`, `train.npy`, `validation.npy`, `test.npy`, and `metadata.json` when `dataset_config.n_pretrain_paths` is positive. The hash depends on the environment name, path counts, episode length, dataset seed, and price-process parameters. Set `dataset_config.force_regenerate: true` to overwrite an existing matching cache. Project configs use `n_pretrain_paths: 5000` while keeping the existing train, validation, and test path counts unchanged.

## Historical Calibration Pipeline

The repository can also calibrate synthetic prices from prepared historical CSVs while keeping calibration and backtesting chronologically separate. Monthly calibration data up to and including `2024-12-31` is used to estimate a log-seasonal calendar-month component. The twelve monthly seasonal values are evaluated as a smooth periodic Fourier curve for daily paths, avoiding hard jumps at month boundaries. Daily calibration data up to the same cutoff is deseasonalized into log residuals, then used to fit an AR(1)/OU residual process and a simple jump component from large AR(1) innovations.

Held-out historical backtesting starts at `2025-01-01`. Those backtest prices are never used for seasonality, residual, OU, or jump calibration. They are only converted into separate rolling-window episodes.

Build calibrated synthetic train/validation/test paths and historical backtest windows:

```bash
PYTHONPATH=src python -m gas_storage_rl.data.build_historical_datasets --config configs/historical_debug.yaml
```

The synthetic splits are stored under `data/cache/{dataset_hash}/`. Historical backtest windows are stored separately under `data/cache/backtest/{dataset_hash}/` as `backtest.npy` plus metadata with each episode start and end date.

Historically calibrated synthetic paths remain log-additive and strictly positive.
They support three environment variants:

- `historical_deterministic`: calibrated monthly log seasonality only
- `historical_ou`: calibrated seasonality plus AR(1)/OU residual noise
- `historical_jump`: calibrated seasonality plus AR(1)/OU residual noise and jumps

## Storage Dynamics

The storage level satisfies `0 <= v_t <= C`. The continuous action space is `Box(-1, 1, shape=(1,))`, where positive actions inject gas and negative actions withdraw gas. Executed actions are clipped by injection and withdrawal rates, storage boundaries, and terminal reachability. After each action, the remaining inventory must stay within the range from which the target terminal inventory can still be reached with the remaining injection and withdrawal rates. The MVP uses efficiency `1`, transaction costs `0`, leakage `0`, and no volume-dependent rates.

## Observation Space

Observations are `np.float32` vectors:

```text
[
  storage_level / capacity,
  price / price_scale,
  sin(day_of_year),
  cos(day_of_year),
  remaining_time,
  target_terminal_inventory / capacity,
]
```

The MVP intentionally does not include M+1 futures features. Calendar features are
derived from the episode start date when date metadata is available.

## Reward

The raw economic cashflow is computed from the executed action:

```text
raw_cashflow_t = -executed_action_t * price_t
```

Buying gas is negative cashflow and selling gas is positive cashflow.

At the final step a terminal inventory penalty is applied:

```text
terminal_penalty = -penalty_factor * mean_training_price * abs(v_T - target_inventory)
raw_reward_t = raw_cashflow_t + terminal_penalty_t
scaled_reward = raw_reward / reward_scale
```

For non-terminal steps, `terminal_penalty_t` is zero. The environment returns the
scaled cashflow reward to RL algorithms and records the unscaled reward as
`raw_reward` in `info`.

## Benchmarks

Implemented benchmarks are:

- `RandomPolicy`: samples uniformly from `[-1, 1]`.
- `RuleBasedPolicy`: buys below the 30 percent training quantile, sells above the 70 percent quantile, and enforces safe liquidation near maturity.
- `LSMCBenchmark`: Least-Squares Monte Carlo with action grid `[-1, -0.5, 0, 0.5, 1]`, a configurable storage inventory grid, terminal-feasible action clipping, and polynomial continuation features.
- `PerfectForesightBaseline`: deterministic upper bound solved with `scipy.optimize.linprog`.
- `OracleClonedPolicy`: neural observation-only policy trained by supervised imitation of perfect-foresight actions from the `pretrain` and `train` splits, then reported only on `validation` and `test`.

## Algorithms

PPO, SAC, and TD3 are created through Stable-Baselines3. RL Baselines3 Zoo may be used as a reference for implementation details and hyperparameters, but the core framework is this repository.

## Setup And Tests

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

CLI run commands print compact progress updates to stderr while keeping final JSON
summaries on stdout.

## Run Experiments

Train one RL agent:

```bash
PYTHONPATH=src python -m gas_storage_rl.training.run_experiment --config configs/debug.yaml --algorithm ppo
```

By default, training is skipped when `runs/` already contains a completed run
with the same effective environment, dataset, price-process, training,
evaluation, agent hyperparameter, seed, and pretraining-policy configuration.
Pass `--rerun` to force a new run:

```bash
PYTHONPATH=src python -m gas_storage_rl.training.run_experiment \
  --config configs/debug.yaml \
  --algorithm ppo \
  --rerun
```

Run a sequential group with `training_config.n_seeds` independent training seeds:

```bash
PYTHONPATH=src python -m gas_storage_rl.training.run_experiment_group \
  --config configs/historical_jump_c200.yaml \
  --algorithm ppo
```

Override the configured group size when needed:

```bash
PYTHONPATH=src python -m gas_storage_rl.training.run_experiment_group \
  --config configs/historical_jump_c200.yaml \
  --algorithm ppo \
  --n-seeds 3
```

For each `seed_index`, the group runner derives deterministic `env_seed` and
`agent_seed` values from `master_seed`. The `dataset_seed` and `eval_seed` remain
constant, so all runs use the same path datasets and fixed validation/test episodes
while varying network initialization, exploration, minibatch sampling, and the sampled
training-episode sequence. The same `(env_seed, agent_seed)` pair is derived for a
given `seed_index` independently of the selected algorithm, which supports paired
PPO/SAC/TD3 experiment groups.

Group metadata is stored under:

```text
runs/experiment_groups/<group_id>/
  group_config.json
  group_metadata.json
  runs.csv
```

## Hyperparameter Tuning

Phase 1 hyperparameter tuning uses Optuna with the TPE sampler. Each trial trains
one algorithm-specific hyperparameter configuration on the train split for seed
indices `0`, `1`, and `2`, and also tunes a shared `reward_scale_multiplier`
from `[0.25, 0.5, 1.0, 2.0, 4.0]`. The effective environment `reward_scale` is
the base config value multiplied by that trial value. HPO selects by the mean
final validation return after the fixed training budget. The test split is not
used by the HPO runner.

```bash
PYTHONPATH=src python -m gas_storage_rl.hpo.run_hpo \
  --config configs/ou_c30.yaml \
  --algorithm ppo \
  --n-trials 32 \
  --seed-indices 0 1 2 \
  --total-timesteps 500000
```

HPO output is stored under:

```text
runs/hpo/<study_id>/
  optuna_study.db
  study_config.json
  search_space.json
  metadata.json
  trials.csv
  trial_seed_runs.csv
  best_trial.json
  best_config.json
```

Trials are valid only when all three tuning-seed runs finish successfully. The
Optuna objective is `mean_validation_return_raw` averaged across the tuning
seeds; trial exports also store the across-seed standard deviation, median, and
minimum validation return. Phase 2 final runs are started manually from
`best_config.json` with disjoint seed indices:

```bash
PYTHONPATH=src python -m gas_storage_rl.training.run_experiment_group \
  --config runs/hpo/<study_id>/best_config.json \
  --algorithm ppo \
  --seed-indices 100 101 102 103 104 105 106 107
```

Those final runs use the validation split for learning curves/AULC and the test
split for final holdout performance.

Each seed still produces a complete normal RL run under `runs/<run_id>/`. If one seed
matches an already completed run and `--rerun` is not set, it is recorded as
`skipped` in `runs.csv`. If one seed fails, the failure is recorded in `runs.csv`
and the remaining seeds continue.

Each RL run writes:

- `config.json`
- `metadata.json`
- `metrics.csv`: completed training episodes
- `evaluations.csv`: periodic validation according to `training_config.eval_freq`
  plus final post-training validation
- `best_validation_model.zip`
- `best_risk_adjusted_validation_model.zip`: highest validation
  `mean_return_raw - risk_adjusted_std_penalty * std_return_raw`
- `final_model.zip`
- `final_summary.json`: final validation metrics plus
  `AULC_validation_return_raw` and `normalized_AULC_validation_return_raw`
- `sb3_logs/`

Experiment-group `runs.csv` also includes the two validation AULC columns for
completed or skipped runs whose summaries contain them. The normalized AULC divides
the validation-return AULC by `training_config.total_timesteps`, which makes it useful
for HPO comparisons across runs with the same validation protocol.

The training command does not evaluate the synthetic test split or historical backtest
split. Run holdout evaluations manually after model selection.

Fine-tune from behavior-cloning pretraining weights:

```bash
PYTHONPATH=src python -m gas_storage_rl.training.run_experiment \
  --config configs/debug.yaml \
  --algorithm ppo \
  --pretrained-policy runs/pretraining/<run_id>/policy_state_dict.pt
```

## Holdout Evaluation

Run manual holdout evaluation on the synthetic test split:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_holdout_evaluation --run-dir runs/<run_id> --split test
```

Write final per-episode RL metrics for comparison plots:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_holdout_evaluation \
  --run-dir runs/<run_id> \
  --split validation \
  --write-final-episode-metrics
```

Run manual holdout evaluation on historical backtest windows:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_holdout_evaluation --run-dir runs/<run_id> --split backtest
```

## Run Benchmarks

Run default benchmarks:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks --config configs/debug.yaml
```

By default, benchmark runs evaluate and log the `train` and `validation` splits only.
The synthetic `test` split is evaluated only when requested explicitly:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks --config configs/debug.yaml --split test
```

Multiple splits can be requested by repeating `--split`. Benchmark outputs are stored
under `runs/benchmarks/<timestamp>-<config_name>-<config_hash>/` with `config.json`,
`metadata.json`, one `benchmark_metrics_<split>.json` file per evaluated split, and a
long-format `benchmark_metrics.csv` with one row per benchmark and split. The fitted
LSMC policy is stored as `lsmc_policy.pkl`; when the oracle-cloned benchmark is
included, its policy is stored as `oracle_cloned_policy.pt`. These artifacts allow
diagnostic comparison plots without refitting benchmark policies. The same runner works
for historically calibrated synthetic price configs because it evaluates the dataset
produced by the selected config.

For learning-curve plots, benchmark metrics can also be expanded onto the same
training-step coordinates as RL `evaluations.csv`. When `training_config.total_timesteps`
and `training_config.eval_freq` are present in the config, this writes
`benchmark_evaluations.csv` at:

```text
0, eval_freq, 2 * eval_freq, ..., total_timesteps
```

These rows are benchmark reference lines, not benchmark training curves. The values are
constant across steps because `random`, `rule_based`, `lsmc`, `perfect_foresight`, and
`oracle_cloned_policy` are evaluated after their respective setup or fit procedure.
`perfect_foresight` should be interpreted as an oracle upper bound because it solves
each requested episode with full future price information.

The timeline can be overridden explicitly:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks \
  --config configs/debug.yaml \
  --split validation \
  --timeline-total-timesteps 100000 \
  --timeline-eval-freq 20000
```

Final per-episode benchmark comparisons can be written with:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks \
  --config configs/debug.yaml \
  --split validation \
  --include-oracle-cloned-policy \
  --write-final-episode-metrics
```

This adds `final_episode_metrics_validation.csv` with one row per method and fixed
validation episode. The synthetic `test` split still runs only when requested
explicitly:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks \
  --config configs/debug.yaml \
  --split test \
  --include-oracle-cloned-policy \
  --write-final-episode-metrics
```

For stochastic random-policy comparisons, pass multiple action seeds:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks \
  --config configs/debug.yaml \
  --split validation \
  --random-policy-seed 1 \
  --random-policy-seed 2 \
  --random-policy-seed 3
```

The random-policy aggregate in `benchmark_metrics.csv` and
`benchmark_evaluations.csv` is then computed over all requested random seeds and all
episodes. `final_episode_metrics_<split>.csv` keeps the seed column so path-level plots
can either show individual random draws or aggregate them later.

LSMC uses `evaluation_config.lsmc_action_grid` and
`evaluation_config.lsmc_n_inventory_levels`. Increasing the number of inventory levels
improves the storage-state coverage of the regression, but increases the fit cost
roughly linearly.

### Oracle Trajectories And Pretraining

Perfect-foresight path-level trajectories can be logged explicitly:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks --config configs/debug.yaml --write-perfect-foresight-trajectories
```

The oracle-cloned neural benchmark is optional. It solves perfect-foresight
trajectories internally for `pretrain` and `train`, fits a small MLP policy on the
resulting observation/action pairs, and adds `oracle_cloned_policy` metrics only for
requested `validation` and `test` splits:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks \
  --config configs/debug.yaml \
  --split validation \
  --split test \
  --include-oracle-cloned-policy
```

For behavior-cloning pretraining, generate only the separate pretrain trajectories:

```bash
PYTHONPATH=src python -m gas_storage_rl.evaluation.run_benchmarks --config configs/debug.yaml --split pretrain --write-perfect-foresight-trajectories
```

This writes `perfect_foresight_trajectories_<split>.jsonl` with one row per price path,
including `start_date`, `end_date`, `prices`, `actions`, `storage_levels`,
`objective_value`, `terminal_deviation`, and `success`. Synthetic generated paths do not
always carry date metadata, so `start_date` and `end_date` are `null` unless the dataset
provides per-path date ranges.

Train a behavior-cloning policy from those trajectories:

```bash
PYTHONPATH=src python -m gas_storage_rl.pretraining.behavior_cloning \
  --config configs/debug.yaml \
  --algorithm ppo \
  --trajectories runs/benchmarks/<run_id>/perfect_foresight_trajectories_pretrain.jsonl
```

The pretraining run writes `pretrained_model.zip`, `policy_state_dict.pt`,
`metadata.json`, and `training_history.json` under
`runs/pretraining/<timestamp>-<config_name>-<algorithm>-<config_hash>/`.
During pretraining, each oracle action sequence is replayed through the storage
environment. The actor is fitted to the requested oracle action, while critic targets
are discounted returns from the environment's scaled rewards. PPO fits its value
function; SAC and TD3 fit both Q-functions against the same return target and then
synchronize their target networks. Set
`pretraining_config.value_loss_coefficient` to control the PPO value-loss weight
(default: `0.5`).

## Plotting

Plotting helpers live in `gas_storage_rl.plotting` and return Matplotlib figures for price paths, action paths, storage levels, cumulative cashflows, price-action scatter, learning curves, and return distributions.

### Single-Run Diagnostics

Create trajectory plots for one saved RL run:

```bash
PYTHONPATH=src python -m gas_storage_rl.plotting.plot_run --run-dir runs/<run_id> --split validation --path-id 0
```

By default, `plot_run` uses `best_validation_model.zip`. Pass `--model final`
to inspect the final training checkpoint or `--model risk_adjusted` for the
risk-adjusted validation checkpoint.

### RL Versus Benchmark Comparisons

Create comparison plots across multiple RL runs and a benchmark run:

```bash
PYTHONPATH=src python -m gas_storage_rl.plotting.plot_comparison \
  --rl-run-dir runs/<ppo_run_id> \
  --rl-run-dir runs/<sac_run_id> \
  --rl-run-dir runs/<td3_run_id> \
  --benchmark-run-dir runs/benchmarks/<benchmark_run_id> \
  --split validation
```

This creates a learning-curve plot with one line per RL run plus benchmark reference
lines, a final return violin plot, and a relative-regret violin plot against
`perfect_foresight` when the required `final_episode_metrics_<split>.csv` files are
available. Relative regret is computed per episode as
`(perfect_foresight_return - method_return) / abs(perfect_foresight_return)`, with a
small numerical floor in the denominator.

Create an on-demand policy diagnostic plot for one fixed path without logging all
step-level validation trajectories:

```bash
PYTHONPATH=src python -m gas_storage_rl.plotting.plot_policy_comparison \
  --benchmark-run-dir runs/benchmarks/<benchmark_run_id> \
  --rl-run-dir runs/<ppo_run_id> \
  --rl-run-dir runs/<sac_run_id> \
  --split validation \
  --path-id 17
```

The diagnostic figure shows spot price, an executed-action heatmap with one row per
method, storage levels, and cumulative raw economic returns for the selected RL policies plus
rule-based, LSMC, oracle-cloned policy when available, and perfect foresight. The
action heatmap uses a fixed scale from withdrawal (`-1`) to injection (`1`). A yellow
overlay on the upper half of a cell indicates that the requested action was clipped
before execution.

### Price Path Plots

Create raw 729-day synthetic price-path plots from the training split:

```bash
python scripts/create_price_path_plots.py --config configs/ou_c30.yaml --n-paths 50
```

The price-path plotting scripts use `--split train` by default. Validation and test
price paths are plotted only when requested explicitly:

```bash
python scripts/create_price_path_plots.py --config configs/ou_c30.yaml --split validation --n-paths 50
python scripts/create_price_path_plots.py --config configs/ou_c30.yaml --split test --n-paths 50
```

Create raw 729-day historically calibrated synthetic price-path plots:

```bash
python scripts/create_historical_price_path_plots.py \
  --config configs/historical_ou_c30.yaml \
  --split train \
  --n-paths 50
```

The plotted environment is taken from the selected config. Raw price-path plots include
a dashed vertical line at simulation day 365, marking the beginning of the second year.
Historical backtest price paths are excluded by default and are plotted only with
`--include-backtest`.
