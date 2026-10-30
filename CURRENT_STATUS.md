# Safe RL Research — Current Status

## Current stage

Experiments 01–05, 06A, 07, and 08 are complete and scientifically reviewed.
Experiment 06/IQL remains unfinished: its full ten-seed study was not launched.
The present task finalizes Experiment 08 and records the research decision history;
no Experiment 09, new collection method, or safety filter is approved here.

The long-term objective remains safe learning with active expert-data acquisition.
The evidence now separates imitation-error detection, actual controller improvement,
and whether an initial state admits any safe control sequence.

## Fixed benchmark and data

- Pendulum-v1; PD expert `clip(-8*theta - 2*omega, -2, 2)` N·m.
- Safety set: `|theta| <= 1 rad`, `|omega| <= 3 rad/s`.
- Normalized margin: `min(1 - |theta|, 1 - |omega|/3)`.
- Experiment 01: 81×81 grid over `[-0.8,0.8] × [-2,2]`, 200 steps;
  3,331/6,561 trajectories safe (50.77%). First violations: 0.10–2.10 s,
  mean 0.4939 s. This original benchmark uses observation-derived wrapped angles
  and pre-action samples; later all-201-raw-state safety results are distinct.
- D0: seed 42, 100×200 = 20,000 transitions from
  `theta0 ∈ [-0.25,0.25]`, `omega0 ∈ [-0.5,0.5]`.
  Mean/best/worst return: −0.220083 / −0.000562 / −0.974336;
  minimum pre-action margin 0.734792; zero violations, including final successors.
  D0 remains unchanged at `data/pendulum_D0.npz`.

## Completed evidence

| Experiment | Main finding | Reviewed source / existing commit |
|---|---|---|
| 01–02: expert map and D0 | PD is not universally safe; nominal demonstrations are safe and narrowly supported. | `experiments/01_expert_map.py`, `experiments/02_generate_D0.py`; `82f8949`, `c5d850c` |
| 03: ensemble disagreement | Useful for detecting imitation error in this setting, with saturation/tie and seed qualifications; this did not test acquisition benefit. | `experiments/03_robustness_analysis.py`; `e037a25` |
| 04: active acquisition | At 5,000 transitions, uncertainty−random return +15.644, 95% CI [−42.176, 59.125]; violation-rate difference −0.200 percentage points [−0.550, 0.000]. No demonstrated primary benefit. Distance improved return with safety tradeoffs. | [Report](docs/experiment_04_report.md); `297828a` |
| 05: rollout horizon | Uncertainty H20−H200 return +8.164 [−39.914, 52.714]; violation-rate difference −0.010 points [−0.090, 0.060]. Near-equilibrium data fell from 89.56% to 14.33%, but no reliable primary improvement; resets increased about tenfold. | [Report](docs/experiment_05_report.md); `92fe329` |
| 06A: counterfactual actions | Safe gains in both return metrics at 38/96 states above 1e−6, 21/96 above 0.01, none above 0.1. No rescues among 40 PD-unsafe states; two alternatives introduced violations. Oracle diagnostics, not a deployed policy. | [Report](docs/experiment_06a_report.md); `ef2d4b6` |
| 07: recoverability optimization | 0/40 PD-unsafe states had verified safe open-loop or tracking witnesses. All 48 converged non-equilibrium reference candidates failed open-loop execution; tracking remained safe on all 16 references. Solver failure alone did not prove nonviability. | [Report](docs/experiment_07_report.md); `040dea5` |
| 08: finite-horizon viability | **40/40 PD-unsafe states independently certified nonviable for the declared real-valued discrete model**, no unresolved PD-unsafe cases; no false certificates on 16 reference controls. | [Report](docs/experiment_08_report.md); commit reference in [decision log](docs/RESEARCH_DECISION_LOG.md) |

Experiment 04 uses paired seeds 42–51; Experiment 05 uses 52–61. Reported intervals
are seed-bootstrap intervals conditional on each study's fixed evaluation set.
Zero-inclusive intervals do not establish equivalence. Both studies retain all
seeds, negative findings, and separate return/safety outcomes.

## Experiment 08 model and scope

The exact real-valued map is
`omega_next = omega + (15*sin(theta) + 3*u)/20`,
`theta_next = theta + omega_next/20`, with `h=1/20`, `g=10`, `m=l=1`,
continuous admissible torque `|u|<=2`, and the closed safety set above.
Initial states are the exact archived binary64 values, not rounded display values.
The horizon is 200 transitions, with safety required at all 201 sampled states.

On the safe strip all state/action derivatives are nonnegative; a step beginning
in the safe set satisfies `|omega_next|<=81/20<8`, so speed clipping is inactive.
A safe-prefix constant −2 trajectory with a strict upper crossing bounds every
hypothetical safe path from below; +2 gives the reflected lower-crossing proof.
Directed-rounding MPFR intervals and an independent exact-rational/Taylor checker
verify the premises and crossing. Opposite exits do not certify nonviability.

All 40 certificates cross an angle boundary (20 upper, 20 lower), at steps 2–23
(0.10–1.15 s), using 128-bit precision. Minimum certified separation is
0.000527185965922 rad. The five reviewed pilot cases were reused unchanged;
the remaining 38 PD-unsafe and 14 reference states completed verification.
All 21 technical tests, 114 direction checks, exact interval repeats, and
171 bitwise simulator replay comparisons passed. No global viability grid was needed.

The 16 references have empirical Gymnasium PD witnesses, not mathematical
viability proofs from this method. Equilibrium has an exact zero-input witness.
No global floating-point implementation-error bound, continuous-time guarantee,
robustness certificate, infinite-horizon viable region, or safety filter is claimed.
The 40 states were selected by prior PD failure; 40/40 is not a population estimate.
Within this model, no admissible controller can prevent their finite-horizon exits.
This conclusion does not retroactively turn Experiment 07 solver statuses into proofs.

## Experiment 06/IQL remains unfinished

The seed-41 expanded-data critic diverged. Freezing 138 safety-stopped continuation
values prevented that catastrophic divergence in the paired diagnostic, but did
not establish accurate values. A smooth physically bounded readout prevented
unbounded predictions while producing extensive saturation and inaccurate
PD-return reference estimates. These were diagnostic-only interventions; they were
not adopted as production IQL or combined. The planned full study remains on hold.

The deterministic PD datasets lack direct counterfactual actions at identical
states. Greater state coverage does not establish action coverage or supported
improvement beyond the teacher. Advantage weighting can reweight imitation;
it does not supply missing counterfactual evidence. IQL source/config/test/report
files remain untracked and excluded from Experiment 08 finalization. All local
pilot/diagnostic artifacts are preserved under `results/offline_rl/`.

## Handoff and preservation

Read [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md), the
[decision log](docs/RESEARCH_DECISION_LOG.md), and the current reports before
proposing further work. Do not restart completed experiments or tune methods using
reviewed results. No next experiment is authorized by this status update.

- Scientific implementation: `src/viability*.py`; entry points:
  `experiments/08_viability_pilot.py` and `experiments/08_viability_study.py`.
- Frozen configuration/dependency files: `configs/experiment_08_*.json` and
  `requirements_experiment_08.txt`; tests: `tests/test_viability*.py`.
- Saved evidence: `results/viability/pilot/` and `results/viability/full_study/`;
  57 state rows, 114 direction rows, exact certificates, identity/source hashes,
  replay arrays, and verification records. Generated artifacts remain outside Git.
- [Reproduction and archive verification](docs/experiment_08_reproduction.md)
  explain dependencies and the historical preservation hashes. The authorized
  context-document updates must be distinguished from changes to scientific artifacts;
  original manifests must not be rewritten to accept new content.
