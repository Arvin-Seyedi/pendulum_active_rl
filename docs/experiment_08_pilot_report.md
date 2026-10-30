# Experiment 08 — Five-case finite-horizon viability pilot

This pilot contains only equilibrium and state IDs 2, 33, 37, and 94. No remaining cohort states or viability grids were evaluated. No optimization, RL training, or expert-training data collection was performed.

## Model and mathematical justification

The declared real-valued discrete map is v = omega + (15 sin(theta) + 3u)/20, theta_next = theta + v/20. Physical parameters, including h=1/20, are exact rationals; each archived binary64 initial value is treated as an exact dyadic rational. Bounds are |theta|<=1, |omega|<=3, |u|<=2, with 200 transitions and 201 sampled states. Initial hex strings, integer ratios, and raw bytes are saved.

For |theta|<=1, cos(theta)>=1/2, so all state/action derivatives of this map are nonnegative. Any transition beginning in the safety set has |omega_next|<=81/20<8; simulator speed clipping is irrelevant to this comparison. A constant -2 path with a wholly safe prefix and a strict upper crossing bounds any hypothetical safe admissible path from below. Constant +2 gives the reflected lower-crossing argument. Thus such a crossing by k<=200 rules out every safe admissible control sequence for the declared model. Opposite-direction exits do not certify nonviability, and no trajectory is propagated after its first exit or ambiguous boundary.

## Pilot results

| State | Group | Real-model result | Certifying torque | Step / time | Certified crossing separation (approx. rad) | Precision |
|---|---|---|---:|---:|---:|---:|
| 96 | equilibrium | exact_equilibrium_viable | — | — | — | — |
| 2 | nominal_safe | no_nonviability_certificate | — | — | — | — |
| 33 | challenging_safe | no_nonviability_certificate | — | — | — | — |
| 37 | PD_unsafe | certified_nonviable | 2 | 13 / 0.65 s | 0.0139789274341 | 128 |
| 94 | PD_unsafe | certified_nonviable | -2 | 6 / 0.30 s | 0.000527185965922 | 128 |

The reported separations are decimal displays of exact saved rational lower bounds. The certified counts apply only to the two PD-unsafe pilot states, not to the full set of 40. Equilibrium has a separate exact zero-torque witness. For states 2 and 33, absence of a certificate is not a mathematical viability proof; their full-horizon PD safety is verified separately in Gymnasium.

| State | Torque | Extremal test status | First exit step | Maximum theta interval width | Maximum omega interval width | Safe-prefix minimum normalized margin lower bound |
|---|---:|---|---:|---:|---:|---:|
| 96 | -2 | opposite_exit | 8 | 2.057e-38 | 8.228e-38 | 0.0743732447735 |
| 96 | 2 | opposite_exit | 8 | 2.057e-38 | 8.228e-38 | 0.0743732447735 |
| 2 | -2 | opposite_exit | 9 | 2.057e-38 | 9.404e-38 | 0.083946235273 |
| 2 | 2 | opposite_exit | 7 | 1.763e-38 | 8.228e-38 | 0.0999963081335 |
| 33 | -2 | opposite_exit | 17 | 9.698e-38 | 3.644e-37 | 0.158260157755 |
| 33 | 2 | opposite_exit | 5 | 1.469e-38 | 7.053e-38 | 0.06770805834 |
| 37 | -2 | opposite_exit | 5 | 2.939e-38 | 8.228e-38 | 0.0338523429631 |
| 37 | 2 | certified_nonviable | 13 | 1.175e-37 | 3.350e-37 | 0.0732511308072 |
| 94 | -2 | certified_nonviable | 6 | 2.351e-38 | 5.290e-38 | 0.0696836670888 |
| 94 | 2 | opposite_exit | 3 | 1.175e-38 | 3.526e-38 | 0.103988994121 |

## Independent checking and simulator comparison

All 17 technical tests pass. All 10 interval records pass an independent checker that imports neither MPFR nor the producer. It evaluates exact rational alternating Taylor bounds for sine, checks enclosure inclusion at every transition, verifies exact initial IDs/values, every safe-prefix box, and the strict crossing direction. It also rejects forged records, wrong directions, ambiguous boundary crossings, unsafe prefixes, and altered model parameters. All generation repeats reproduce the exact interval records.

MPFR directed rounding uses integer division by 20 rather than a rounded 0.05 literal. Endpoints are serialized as exact numerator/denominator strings. Precision escalation is fixed at 128, 256, then 512 bits only for boundary ambiguity; numerical tolerances never relax safety.

| State | Archived PD return | Archived PD minimum raw margin | PD violation | New recorded-action replay |
|---|---:|---:|---|---|
| 96 | 0.000000000 | 1.000000000 | False | bitwise equal |
| 2 | -0.028573762 | 0.925424177 | False | bitwise equal |
| 33 | -0.899955116 | 0.729121348 | False | bitwise equal |
| 37 | -138.354221948 | -5.283185292 | True | bitwise equal |
| 94 | -819.710731644 | -4.524049346 | True | bitwise equal |

All ten extremal Gymnasium trajectories were evaluated over the full 200 steps using executed float32 torques and all 201 raw states, and independently replayed bitwise. Model and simulator first-exit steps match for 10/10 directions. Maximum observed prefix discrepancies bounded using the real-model enclosures are 3.62662e-16 rad and 1.41426e-15 rad/s. These are observations at the tested prefixes, not global implementation-error bounds.

Certificates concern the declared real map, not NumPy/libm floating-point semantics. At u=+/-2 the torque product is exact, but binary 0.05, operation rounding, and sine implementation remain different. No justified implementation-wide error envelope is claimed. Bitwise simulator agreement does not upgrade a model certificate to a machine-level or physical-system proof.

## Cost, preservation, and reproduction

Scientific interval propagation: 81 steps; deterministic generation repeat: 81 steps; independent rational checker: 81 checks. Generation took 0.000815 s, repetition 0.000622 s, and rational checking 0.090865 s. Unit-test costs are separate from these five-case measurements.
New Gymnasium interactions: 5000 steps and 25 resets, comprising five archived PD-action replays, ten extremal executions, and ten extremal replays. No new sampled initial states or training transitions were added. Numerical library versions: {'gmpy2': '2.3.2', 'GMP': 'GMP 6.3.0', 'MPFR': 'MPFR 4.2.2', 'MPC': 'MPC 1.4.1'}.

SHA-256 checks preserve 11708 pre-existing files, including Experiments 01–07, D0, the complete Experiment 07 pilot/full-study artifacts, and unfinished IQL diagnostics. The scientific configuration and source snapshots are saved with all certificates, noncertifying attempts, independent checks, and replay diagnostics under results/viability/pilot/.

Install requirements_experiment_08.txt in a separate environment. Run `python -m unittest discover -s tests -p "test_viability*.py" -v`, then `python experiments/08_viability_pilot.py --pilot`. Existing outputs require explicit `--resume`; `--verify` only verifies completed hashes and never launches computations. Source Experiment 07 artifacts are required. Generated reports remain in results; the reviewed documentation is not overwritten by reproduction.

These are sampled-time statements over the declared 200-step horizon, not continuous-time, infinite-horizon, or real-world safety claims. No conclusion is assigned to the other 38 PD-unsafe states or the other reference states before approval to proceed.
