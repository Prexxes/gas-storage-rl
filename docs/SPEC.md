# Mathematical Specification

## Price Processes

All environments use log-additive synthetic gas spot prices:

```text
log_price_t = seasonal_log_price_t + ou_residual_t + jump_component_t
price_t = exp(log_price_t)
```

The deterministic environment sets both stochastic terms to zero. The OU environment includes a mean-reverting residual. The jump environment adds sparse regular jumps and larger stress jumps. Historical calibration is intentionally out of scope for the MVP; configurable fallback parameters define the synthetic processes.

## Storage Dynamics

For capacity `C`, inventory `v_t`, and requested action `u_t`, the executed action is:

```text
a_t = clip(u_t, max(-withdrawal_rate, -v_t), min(injection_rate, C - v_t))
v_{t+1} = v_t + a_t
```

The MVP sets efficiency to `1`, transaction costs to `0`, leakage to `0`, initial inventory to `0`, and target terminal inventory to `0`.

## Reward

Raw cashflow is:

```text
cashflow_t = -a_t p_t
```

At the final decision step:

```text
lambda_terminal = penalty_factor * mean_training_price
terminal_penalty = -lambda_terminal * abs(v_T - target_inventory)
```

The raw reward is cashflow plus the terminal penalty if final. The scaled reward returned to the RL algorithm is `raw_reward / reward_scale`.

## Observation Normalization

The observation vector is:

```text
[v_t / C, p_t / price_scale, t / (T - 1)]
```

The observation dtype is `np.float32`.

## Benchmarks

Random policy samples uniformly from `[-1, 1]`. Rule-based policy uses training price quantiles with liquidation feasibility checks. LSMC fits continuation values backward on training paths using the discrete grid `[-1, 0, 1]`. Perfect foresight solves a deterministic linear program for each known path.

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

Periodic validation runs after `total_training_env_steps` reaches each configured `eval_freq` interval. The best validation model is saved separately from the final model. Test evaluation is performed after training completes and does not update the policy.

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
