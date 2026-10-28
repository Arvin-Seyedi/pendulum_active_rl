# Experiment 06A — Counterfactual Action Feasibility

Simulation-only diagnostic. No controller was trained, no safety filter was introduced, and no diagnostic transition was added to a training dataset.

## Fixed protocol

32 nominal states uniformly sampled from [-0.25,0.25] × [-0.5,0.5] (seed 606201); 64 challenging states uniformly sampled from [-0.8,0.8] × [-2,2] excluding that nominal rectangle (seed 606202); one equilibrium technical control, excluded from sampled-state summaries. States were frozen before simulation.

Offsets {-0.4,-0.2,0,+0.2,+0.4} N·m were added to the already clipped PD action and deduplicated after clipping and float32 conversion. Each branch used the identical float64 physical initial state and reset seed 606203 + state ID. One candidate action was followed by the unchanged PD controller for 199 steps. Discount 0.99; 200 steps (10 seconds); no stopping at safety crossings. All 201 raw states were checked against |theta| ≤ 1 and |omega| ≤ 3.

## Safe alternatives improving both return metrics

A hit requires a distinct action, minimum raw margin ≥ 0, and both return gains strictly greater than the stated threshold. Percentages are fractions of states, not fractions of action proposals.

| Group | States | PD violations | >1e-6 | >0.01 | >0.1 |
|---|---:|---:|---:|---:|---:|
| all_sampled | 96 | 40 | 38/96 (39.58%) | 21/96 (21.88%) | 0/96 (0.00%) |
| nominal | 32 | 0 | 29/32 (90.62%) | 16/32 (50.00%) | 0/32 (0.00%) |
| challenging | 64 | 40 | 9/64 (14.06%) | 5/64 (7.81%) | 0/64 (0.00%) |
| saturated | 55 | 40 | 0/55 (0.00%) | 0/55 (0.00%) | 0/55 (0.00%) |
| unsaturated | 41 | 0 | 38/41 (92.68%) | 21/41 (51.22%) | 0/41 (0.00%) |
| PD_safe | 56 | 0 | 38/56 (67.86%) | 21/56 (37.50%) | 0/56 (0.00%) |
| PD_violating | 40 | 40 | 0/40 (0.00%) | 0/40 (0.00%) | 0/40 (0.00%) |

95% percentile intervals use 10,000 state-level bootstrap resamples within nominal/challenging strata (seed 606204). These describe sampled-state sensitivity, not safety guarantees; degenerate zero-hit intervals do not establish impossibility.

- Joint-gain threshold 1e-06: 0.395833; 95% interval [0.333333, 0.458333].
- Joint-gain threshold 0.01: 0.218750; 95% interval [0.145833, 0.291667].
- Joint-gain threshold 0.1: 0.000000; 95% interval [0.000000, 0.000000].

## Safety and negative findings

| Group | Rescue: actions / states | Both returns improve but PD and alternative remain unsafe: actions / states | New violation: actions / states |
|---|---:|---:|---:|
| all_sampled | 0 / 0 | 25 / 17 | 2 / 2 |
| nominal | 0 / 0 | 0 / 0 | 0 / 0 |
| challenging | 0 / 0 | 25 / 17 | 2 / 2 |
| saturated | 0 / 0 | 25 / 17 | 2 / 2 |
| unsaturated | 0 / 0 | 0 / 0 | 0 / 0 |

| Group | PD mean / worst return | PD mean discounted return | PD minimum raw margin | All-action minimum raw margin |
|---|---:|---:|---:|---:|
| nominal | -0.204698 / -0.761101 | -0.193720 | 0.757380 | 0.749058 |
| challenging | -194.754739 / -899.305647 | -108.227980 | -5.283185 | -5.283185 |
| saturated | -226.454575 / -899.305647 | -125.772844 | -5.283185 | -5.283185 |
| unsaturated | -0.386634 / -2.411386 | -0.372764 | 0.335935 | 0.335935 |

Every action’s paired gains, minimum raw margin, first-crossing step/time (null when censored), and unsafe-state fraction are in `episode_metrics.csv`. `summary.json` includes all stratum intersections, fixed-offset paired summaries and intervals, and cases improving only one return metric. `state_metrics.csv` records per-state eligibility and oracle results.

## Oracle diagnostics

The per-state oracle chooses the distinct, empirically safe alternative with the highest discounted return, if one exists. Both reported gains use that same action; states without a safe alternative remain missing. This uses rollout outcomes and is not deployable policy performance. Existence counts above do not require this oracle choice to maximize the undiscounted return.

| Group | Available / missing states | Mean / minimum / maximum paired return gain | Mean / minimum / maximum paired discounted gain |
|---|---:|---:|---:|
| nominal | 32 / 0 | 0.008749 / -0.038308 / 0.022467 | 0.007913 / -0.034927 / 0.020734 |
| challenging | 24 / 40 | -0.106033 / -1.301193 / 0.023009 | -0.087477 / -1.047685 / 0.021456 |
| saturated | 15 / 40 | -0.180616 / -1.301193 / -0.012697 | -0.150222 / -1.047685 / -0.010698 |
| unsaturated | 41 / 0 | 0.010839 / -0.000997 / 0.023009 | 0.009929 / -0.000369 / 0.021456 |

Across qualifying distinct alternatives, minimum raw safety margin was 0.335935; joint-improvement undiscounted gains ranged from 0.00014800 to 0.023009, and discounted gains from 0.00014002 to 0.021456.

## Verification and interaction costs

- 373 distinct executions including PD and equilibrium; 276 distinct alternatives; 112 duplicate proposals avoided per pass.
- Distinct-action counts per state: {'3': 55, '4': 2, '5': 40}.
- Main pass: 373 resets and 74600 transitions. Replay: 374 resets and 74800 transitions. Total: 747 resets and 149400 transitions.
- Main simulator wall time: 0.723 s; replay simulator wall time: 0.747 s. File hashing, tests, analysis, and plotting are excluded from these simulator timings.
- All technical checks passed: True. Every unique action has a bitwise replay. All 97 PD branches match the preserved evaluator numerically, including bitwise-identical physical states, rewards, and margins. The original evaluator emits -0.0 at the equilibrium first action while the prescribed offset menu emits +0.0; this sole sign-bit exception is documented, and an extra direct equilibrium replay is bitwise identical. It costs one extra reset and 200 steps, included above. Full-horizon/raw-safety checks and equilibrium reward-ceiling checks passed.
- An independent audit recomputed paired returns, thresholds, raw margins, crossing/censoring fields, safety categories, and replay equality directly from the saved arrays. It passed without additional simulator interactions; the largest difference from an independent discounted-sum reduction was below 1e-10.
- SHA-256 preservation checks passed for 6408 pre-existing files, including all prior datasets, results, checkpoints, and source files. No prior experiment was rerun.

## Interpretation and limitations

The diagnostic found small, empirically safe local improvements, concentrated entirely in initially unsaturated states: 38/41 unsaturated states had a qualifying alternative, compared with 0/55 saturated states. Of 96 sampled states, 38 (39.58%) admitted gains above 1e-6 in both metrics and 21 (21.88%) admitted gains above 0.01; none exceeded 0.1 in both. The maximum qualifying gains were 0.023009 undiscounted and 0.021456 discounted. These findings support the existence of modest local action improvements, not a demonstrated controller improvement.

The safety findings are negative: none of the 40 PD-violating states was rescued, despite 25 alternatives at 17 such states improving both returns. Across those failing states, 29 alternatives improved undiscounted return and 25 improved discounted return. Two alternatives created new violations at previously safe challenging states. Both reduced the magnitude of a saturated PD action from 2 to 1.6 N·m: state 49 crossed at step 30 (1.50 s), and state 54 at step 35 (1.75 s). Their undiscounted return changes were -136.178162 and -136.792356, respectively. PD violations in the fixed sample first occurred between 0.10 and 1.15 s (mean 0.46 s).

At saturation, the clipped menu can only retain PD or reduce its torque magnitude; it cannot test a larger torque in the saturated direction. Thus the contrast between saturated and unsaturated states does not isolate saturation as a cause and does not establish action optimality outside this menu. Minimum margins near -5.283185 reflect the unwrapped raw angle, including trajectories approaching an equilibrium one full rotation away; they were not replaced by canonical-angle margins.

Positive hits establish local counterfactual action opportunities under this fixed simulator, menu, PD continuation, and finite horizon. Return improvements on violating trajectories do not establish safe improvement. An absence of rescues does not establish that the unsafe states are unrecoverable under other actions or policies.

The 32/64 mixture is a diagnostic sampling design, not a natural operating distribution. Saturation and PD-failure subgroups are descriptive; their sample counts differ. Exact replays are not independent observations. Bootstrap intervals are pointwise, without multiple-comparison correction. Very small positive gains can exceed the numerical threshold without practical importance.

A one-step change followed by PD does not prove multi-step policy improvement. Repeated intervention would change future state visitation and safety. Discounted 200-step returns are not infinite-horizon IQL values. Best-of-menu outcomes use privileged outcome information. Deterministic simulator resets do not establish robustness to model error, noise, or real systems. PD is not a certified backup controller, and no observed violation over sampled discrete steps is not a formal safety guarantee.

All diagnostic trajectories remain isolated under `results/offline_rl/experiment_06A/`. Existing training datasets and IQL implementations remain unchanged.

## Reproduction from the committed files

Use CPython 3.13.2 and the pinned dependencies in `requirements_experiment_06a.txt` (tested on macOS; the process lock also supports Linux). From a fresh checkout:

```sh
python3 -m pip install -r requirements_experiment_06a.txt
python3 -m unittest discover -s tests -p 'test_counterfactual.py' -v
python3 experiments/06a_counterfactual_actions.py --run
```

The runner requires no D0, acquired datasets, checkpoints, saved results, or uncommitted IQL files. Its reusable helpers come from committed Experiments 01–05. Torch is imported by those existing helpers; no network is trained. Configuration and seeds are fixed in `configs/experiment_06a.json`. Generated reports stay in the ignored results directory; this reviewed report is preserved.

An existing output directory requires `--run --resume`. Completed matching artifacts are verified without rerunning trajectories; incompatible source/configuration hashes fail rather than overwrite results. Wall times, protected-file counts, and repository metadata vary by checkout. Bitwise agreement was verified in the recorded environment, not asserted across arbitrary platforms or library versions.

Finalization validation used an isolated checkout containing only committed Experiments 01–05 and the six Experiment 06A files. All 29 tests passed. A full reproduction matched the original state/action manifest and trajectory arrays bitwise, and matched both metric CSV files and `summary.json` byte-for-byte. Completed-run resume performed no simulations or artifact rewrites. This separate reproduction used 149,400 simulator steps and 747 resets; it did not add independent scientific observations or alter the reviewed study artifacts.
