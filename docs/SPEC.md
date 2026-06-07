# Mathematical Specification

## Price Processes

All environments use log-additive synthetic gas spot prices:

```text
log_price_t = seasonal_log_price_t + ou_residual_t + jump_component_t
price_t = exp(log_price_t)
```

The deterministic environment sets both stochastic terms to zero. The OU environment includes a mean-reverting residual. The jump environment adds sparse regular jumps and larger stress jumps. Configurable fallback parameters define the default synthetic processes.

## Historical Calibration

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

For capacity `C`, inventory `v_t`, and requested action `u_t`, the executed action is:

```text
a_t = clip(u_t, max(-withdrawal_rate, -v_t), min(injection_rate, C - v_t))
v_{t+1} = v_t + a_t
```

The MVP sets efficiency to `1`, transaction costs to `0`, leakage to `0`, initial inventory to `0`, and target terminal inventory to `0`.

## Reward

Raw economic cashflow is:

```text
cashflow_t = -a_t p_t
```

Training uses a mark-to-market shaped reward. For non-terminal steps:

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

At the final decision step:

```text
lambda_terminal = penalty_factor * mean_training_price
terminal_penalty = -lambda_terminal * abs(v_T - target_inventory)
shaped_reward_T = cashflow_T + terminal_penalty - v_T p_T
```

The scaled reward returned to the RL algorithm is
`shaped_reward / reward_scale`. Economic evaluation uses cashflow plus terminal
penalty, not the shaped training reward.

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

Periodic validation runs after `total_training_env_steps` reaches each configured `eval_freq` interval. The best validation model is saved separately from the final model. Test and historical backtest evaluation are not part of the training command. They are run manually after model and hyperparameter selection with `gas_storage_rl.evaluation.run_holdout_evaluation`, so holdout results do not influence training-time decisions.

## AULC

Sample efficiency is measured with area under the validation learning curve:

```text
AULC = integral validation_return(step) d step
```

The implementation uses trapezoidal integration over evaluation checkpoints.

## Reproducibility

A master seed should derive dataset, environment, agent, evaluation, and plotting seeds. Dataset splits use disjoint deterministic seeds. For a given environment and capacity, all algorithms train on the same train paths and are evaluated on the same validation and test paths.

## Price Dataset Cache

Price paths are persisted under `data/cache/{dataset_hash}/` when `dataset_config.use_cache` is enabled. The dataset hash is computed from the price-relevant configuration: environment name, episode length, split sizes, dataset seed, and price-process parameters. Each cache directory contains:

```text
metadata.json
train.npy
validation.npy
test.npy
```

`force_regenerate` can be enabled to overwrite a matching cache intentionally. Cached paths are generated data and are excluded from git.
