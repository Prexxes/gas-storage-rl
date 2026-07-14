# Mathematical Specification

## Price Processes

The three non-historical environments use additive synthetic prices:

```text
price_t = 2 + sin(2 * pi * t / 365) + X_t + J_t
```

The deterministic environment sets `X_t` and `J_t` to zero. The OU environment
uses an exactly discretized Ornstein-Uhlenbeck process with speed of mean
reversion `1.0`, long-term mean `0.0`, volatility `1.2`, and initial value `0.0`.
One daily transition uses a time step of `1 / 365`, so the process evolves
smoothly over an annual horizon.
The jump environment adds sparse regular jumps and larger stress jumps. Synthetic
prices may be negative by design. Configurable fallback parameters define the
default synthetic processes.

## Historical Calibration

Historically calibrated environments remain separate from the additive synthetic
benchmark. They use log-additive prices and are strictly positive:

```text
log_price_t = seasonal_log_price_t + ou_residual_t + jump_component_t
price_t = exp(log_price_t)
```

The historical pipeline enforces a chronological split. Calibration inputs must end on or before `2024-12-31`; historical backtest inputs must start on or after `2025-01-01`. Backtest observations are never used to estimate the synthetic data-generating process.

Monthly calibration prices estimate a calendar-month log seasonality:

```text
s_m = mean(log(monthly_price) - mean(log(monthly_price)) | month = m)
```

The monthly adjustments are normalized to zero mean. For daily evaluation, the
default historical pipeline fits a smooth periodic Fourier curve to the twelve
monthly adjustments. This keeps the monthly calibration interpretation while
removing artificial step changes at month boundaries. Daily calibration prices
are converted to residuals:

```text
r_t = log(daily_price_t) - mean_log_price - s_month(t)
```

The centered residuals fit a discrete AR(1) process:

```text
r_t = phi r_{t-1} + epsilon_t
mean_reversion = 1 - phi
volatility = std(epsilon_t without classified jumps)
```

Large innovations, identified by a configurable robust sigma threshold, define the simple jump component. The calibrated synthetic generator then combines the monthly log seasonality, AR(1) residuals, and sparse additive log jumps. Train, validation, and test paths are generated from deterministic disjoint seeds and cached independently from historical backtest windows.

Historical backtest episodes are rolling windows over the held-out backtest CSV. Their cache metadata stores the start and end date of every window so evaluations remain traceable to the original historical period.

## Storage Dynamics

For capacity `C`, inventory `v_t`, requested action `u_t`, remaining decision steps
after the action `r_t`, and target terminal inventory `v*`, the executed action is
first bounded by physical rate and inventory constraints:

```text
physical_lower = max(-withdrawal_rate, -v_t)
physical_upper = min(injection_rate, C - v_t)
```

The action is also bounded so that `v_{t+1}` remains in the corridor from which
`v*` can still be reached:

```text
min_reachable_level = max(0, v* - r_t injection_rate)
max_reachable_level = min(C, v* + r_t withdrawal_rate)
terminal_lower = min_reachable_level - v_t
terminal_upper = max_reachable_level - v_t
a_t = clip(
    u_t,
    max(physical_lower, terminal_lower),
    min(physical_upper, terminal_upper),
)
v_{t+1} = v_t + a_t
```

If a manually constructed state is already outside the terminal-feasible corridor
and the combined bounds are empty, the physical rate and capacity constraints remain
binding.

The MVP sets efficiency to `1`, transaction costs to `0`, leakage to `0`, initial inventory to `0`, and target terminal inventory to `0`.

## Reward

Raw economic cashflow is:

```text
cashflow_t = -a_t p_t
```

At the final decision step, a terminal inventory penalty is applied:

```text
lambda_terminal = penalty_factor * mean_training_price
terminal_penalty = -lambda_terminal * abs(v_T - target_inventory)
reward_t = cashflow_t + terminal_penalty_t
```

For non-terminal steps, `terminal_penalty_t` is zero. The scaled reward returned to
the RL algorithm is `reward_t / reward_scale`.

## Observation Normalization

The observation vector is:

```text
[v_t / C, p_t / price_scale, sin(day_of_year), cos(day_of_year),
 (T - 1 - t) / (T - 1), v_target / C]
```

The calendar features use the path's actual date when date metadata is
available. Synthetic paths without date metadata start on January 1. The
observation dtype is `np.float32`.

Each episode starts with an inventory fraction drawn from a clipped normal
distribution with mean `0.30` and standard deviation `0.05`. The episode target
inventory equals that sampled initial inventory. Training environments draw a
new value on every unconstrained reset. Pretraining and LSMC training use fixed
stratified normal quantiles per path; validation, test, and historical backtest
evaluation use fixed seeded samples. Perfect foresight, LSMC, and policy
evaluation receive the same per-episode initial and target inventories.

Synthetic datasets store contiguous raw paths of length `2T - 1`. Training
environments sample a start offset uniformly at each reset and expose the next
`T` consecutive prices. Pretraining, validation, test, and benchmark runs use
fixed seeded start offsets per path. LSMC continuation regressions include
calendar sine and cosine terms, the target inventory, and interactions with
normalized price and current inventory.

## Benchmarks

Random policy samples uniformly from `[-1, 1]`. Rule-based policy uses training price quantiles with liquidation feasibility checks. LSMC fits continuation values backward on training paths using the discrete grid `[-1, 0, 1]`. Perfect foresight solves a deterministic linear program for each known path. The optional oracle-cloned policy trains a small observation-only neural policy on perfect-foresight actions from the `pretrain` and `train` splits, then reports the cloned policy only on `validation` and `test`.

## Perfect Foresight LP

For path prices `p_t`, variables are actions `a_t`, storage levels `v_t`, and terminal deviation `d`. The minimization objective is:

```text
minimize sum_t a_t p_t + lambda_terminal d
```

Subject to:

```text
v_0 = initial_inventory
v_{t+1} = v_t + a_t
0 <= v_t <= C
-withdrawal_rate <= a_t <= injection_rate
d >= v_T - target_inventory
d >= target_inventory - v_T
```

The negative optimum is the full-information valuation upper bound.

## LSMC

The MVP LSMC benchmark evaluates feasible actions on the grid `[-1, 0, 1]`, regresses continuation values with normalized storage, normalized price, time fraction, capacity information, and degree-two polynomial terms, then evaluates the fitted policy on validation or test paths.

## Metrics

Validation and test metrics include mean, median, standard deviation, minimum, maximum, interquartile mean raw return, terminal deviation, cumulative cashflow, terminal penalty, and clipped-action counts.

Training logs contain one row per completed training episode in `metrics.csv`. Stable-Baselines3 diagnostics are written to `sb3_logs/progress.csv` and available diagnostics such as PPO KL divergence, clip fraction, entropy loss, policy-gradient loss, value loss, SAC/TD3 actor loss, critic loss, entropy coefficient, and learning rate are copied into training rows when SB3 exposes them during callbacks.

Periodic validation runs after `total_training_env_steps` reaches each configured `eval_freq` interval. After the final policy update, one additional validation row is appended to `evaluations.csv`, even when `total_timesteps` is exactly divisible by `eval_freq`. The best validation model is saved separately from the final model. A risk-adjusted validation model is also saved by maximizing `mean_return_raw - risk_adjusted_std_penalty * std_return_raw`, where `risk_adjusted_std_penalty` defaults to `0.5`. Test and historical backtest evaluation are not part of the training command. They are run manually after model and hyperparameter selection with `gas_storage_rl.evaluation.run_holdout_evaluation`, so holdout results do not influence training-time decisions.

## Hyperparameter Tuning

Phase 1 hyperparameter tuning is implemented by `gas_storage_rl.hpo.run_hpo`.
It uses Optuna TPE with local SQLite storage at
`runs/hpo/<study_id>/optuna_study.db`. HPO optimizes algorithm-specific
Stable-Baselines3 hyperparameters for PPO, SAC, or TD3 and a shared
`reward_scale_multiplier` sampled from `[0.25, 0.5, 1.0, 2.0, 4.0]`. The
effective environment `reward_scale` for a trial is the base config
`reward_scale` multiplied by that value. Price-process settings,
train/validation/test split sizes, reward definition, storage restrictions,
capacity, terminal target, benchmark definitions, evaluation metrics, seed
counts, and the fixed training budget are treated as experimental design and are
not tuned.

Each HPO trial trains the suggested hyperparameter configuration on the train
split for seed indices `0`, `1`, and `2`. The `dataset_seed` remains constant;
`env_seed` and `agent_seed` are deterministically derived from `master_seed` and
the seed index. Each seed run contributes the maximum validation
`mean_return_raw` observed in `evaluations.csv`, matching the
`best_validation_model` checkpoint selection rule. The trial objective is the
mean of those selected validation returns across the three seed runs after a
fixed training budget, for example `500_000` timesteps. A trial is valid only if
all three seed runs complete successfully. Trial exports additionally store
across-seed standard deviation, median, minimum validation return, the selected
validation step, and the final validation return.

Passing `--n-jobs N` runs up to `N` Optuna trials in parallel against local
SQLite storage with an extended lock timeout. During optimization, each trial
writes its own `trial_XXXX.json`; aggregate `trials.csv` and
`trial_seed_runs.csv` exports are rebuilt from those trial artifacts after
optimization finishes.

Interrupted studies can be resumed with `--resume-study-dir runs/hpo/<study_id>`.
Resume mode reuses the existing `optuna_study.db`, `study_id`, and per-trial
artifacts. In this mode, `--n-trials` is interpreted as the target total number
of finished Optuna trials; the runner starts only the missing number of trials
and then rewrites `trials.csv`, `trial_seed_runs.csv`, `best_trial.json`, and
`best_config.json`.

HPO never evaluates the test split. Phase 2 final runs are started manually from
the saved `best_config.json` with disjoint seed indices, for example `100..107`,
using `gas_storage_rl.training.run_experiment_group --seed-indices`. Those final
runs train on the train split, use validation for learning curves and AULC, and
evaluate the test split only as the final holdout performance.

## AULC

Sample efficiency is measured with area under the validation learning curve:

```text
AULC = integral validation_return(step) d step
normalized_AULC = AULC / total_timesteps
```

The implementation uses trapezoidal integration over `mean_return_raw` evaluation
checkpoints in `evaluations.csv`. If multiple validation rows share the same
`total_training_env_steps`, the last row is used, so final post-training validation
replaces a callback validation at the same step. `final_summary.json` stores
`AULC_validation_return_raw` and `normalized_AULC_validation_return_raw` under
`validation`; experiment-group `runs.csv` copies both columns for HPO and group
comparison.

## Reproducibility

A master seed should derive dataset, environment, agent, evaluation, and plotting seeds. Dataset splits use disjoint deterministic seeds. For a given environment and capacity, all algorithms train on the same train paths and are evaluated on the same validation and test paths.

## Deterministic overfitting diagnostic

Before long experiment groups, `gas_storage_rl.training.run_overfit_check` can
verify that PPO, SAC, and TD3 can intentionally memorize one fixed episode. The
episode contains 20 alternating low/high prices:

```text
10, 30, 12, 28, 8, 35, 15, 25, 9, 32,
11, 27, 7, 40, 14, 24, 6, 38, 13, 29
```

Capacity, injection rate, and withdrawal rate are one, and both initial and
target inventory are zero. Perfect foresight therefore alternates injection and
withdrawal and earns a raw return of `203`. Training and evaluation deliberately
use the same single path; this is a memorization and training-pipeline check, not
a generalization result.

The default diagnostic trains each algorithm for 100,000 environment steps
(5,000 episodes) with three seeds. At step zero and every 2,500 steps it pauses
training and runs one deterministic, update-free evaluation episode. A seed
passes when it reaches at least 90% of the oracle return with terminal deviation
at most 0.01. An algorithm passes when at least two of three seeds pass.

```bash
python -m gas_storage_rl.training.run_overfit_check \
  --config configs/sanity_overfit.yaml \
  --algorithms ppo sac td3
```

Reports, deterministic actions, evaluation curves, and best/final models are
stored below `runs/sanity_overfit/`. The short pytest coverage only verifies the
runner and its invariants; it does not impose stochastic RL convergence on CI.

## Observation Ablation Diagnostic

Observation ablations keep the environment observation shape fixed at six
features:

```text
[inventory / capacity, price / price_scale, sin(day), cos(day),
 remaining_time, target_inventory / capacity]
```

Disabled features are replaced by neutral constants rather than removed from the
vector. Inventory, price, remaining time, and target inventory are set to `0`;
calendar is set to `(sin, cos) = (0, 1)`. This keeps SB3 policy architectures
and saved-model interfaces comparable across ablations.

The frozen deterministic diagnostic uses four 90-step episodes with fixed
start dates and initial inventories. Prices are deterministic seasonal paths
with a short cycle and fixed local spikes/dips. Training and validation use the
same four episodes for the overfit-style sanity check, and each validation
episode is scored against a perfect-foresight reference with a practically hard
terminal target, matching the environment's terminal-feasibility clipping.

Default variants are `full`, `price_inventory_only`, `no_calendar`,
`no_remaining_time`, `no_target_inventory`, `no_price`, and `no_inventory`.
Evaluation is deterministic and update-free at step zero, periodically during
training, and once after the final update.

```bash
python -m gas_storage_rl.training.run_observation_ablation \
  --config configs/sanity_observation_ablation.yaml \
  --algorithms ppo sac td3
```

Reports, evaluation curves, and final models are stored below
`runs/observation_ablation/`.

Training commands skip duplicate completed runs by default. Duplicate detection compares the effective run configuration that affects training and validation, including environment settings, dataset settings, price-process parameters, training settings, algorithm hyperparameters, seeds, and pretraining policy. Organizational metadata such as `experiment_group_id` and log directory paths does not affect duplicate detection. Passing `--rerun` forces a new timestamped run.

## Price Dataset Cache

Price paths are persisted under `data/cache/{dataset_hash}/` when `dataset_config.use_cache` is enabled. The dataset hash is computed from the price-relevant configuration: environment name, episode length, split sizes, dataset seed, and price-process parameters. Each cache directory contains:

```text
metadata.json
train.npy
validation.npy
test.npy
```

`force_regenerate` can be enabled to overwrite a matching cache intentionally. Cached paths are generated data and are excluded from git.
