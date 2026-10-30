# Experiment 08 — Finite-Horizon Safety Viability Analysis

Primary outcome: **40/40 PD-unsafe initial states have independently verified nonviability certificates** for the declared real-valued discrete model. Unresolved PD-unsafe states: 0 ([]). This result is separate from Gymnasium execution and does not constitute a real-world impossibility proof.

## Frozen protocol and scope

The fixed cohort contains 40 PD-unsafe states, 16 PD-safe references (eight nominal and eight challenging), and equilibrium. The five reviewed pilot cases were reused without recomputation; 52 remaining states were evaluated. Initial IDs, binary64 bit patterns, and both extremal torques remain unchanged. The model is v = omega + (15 sin(theta) + 3u)/20 and theta_next = theta + v/20, with exact rational parameters, |theta|<=1, |omega|<=3, |u|<=2, and horizon 200 transitions (201 sampled states).

On the safe angle strip, cos(theta)>=1/2 and every state/action derivative is nonnegative. A constant -2 path with a safe prefix and a strict upper crossing bounds every hypothetical safe admissible path from below; constant +2 provides the reflected lower-crossing certificate. A certified exit by k<=200 therefore rules out all safe admissible sequences through 200 steps. All prefix interval boxes must remain inside the safety set. Opposite exits and boundary overlap never imply nonviability. Speed clipping is inactive on a step starting in the safety set because |omega_next|<=81/20<8.

MPFR directed rounding uses exact integer division by 20. The independent checker imports neither MPFR nor the producer: it uses exact Fraction arithmetic and alternating Taylor bounds for sine to check every transition enclosure, identity, prefix, and strict crossing. Precision escalation remains 128, 256, then 512 bits only for boundary ambiguity. Exact rational endpoints and separations are preserved; decimal summaries are displays.

## Outcomes and numerical diagnostics

Certificate directions: {'upper': 20, 'lower': 20}; components: {'theta': 40}. Certified crossing steps range from 2 to 23, or 0.1–1.15 s. Final precision counts across all extremal directions: {'128': 114}.
Certified separation displays range from 0.000527185965922 to 0.10908001764 (in the crossed component's units). Maximum certificate interval widths are 4.995851e-37 rad and 1.469368e-36 rad/s. The smallest certified-prefix normalized-margin lower bound is 0.0102410550863.

False nonviability classifications among the 16 PD-safe reference controls: **0**. Their PD trajectories remain simulator-safe in 16/16 cases. No certificate for a control is not itself a mathematical viability proof. Equilibrium has the separate exact u=0 witness. All noncertifying directions are retained; final direction statuses are {'opposite_exit': 74, 'certified_nonviable': 40}.

Per-state certification status, direction, crossing step/time, separation, interval width, and precision appear in state_results.csv and state_details.md. direction_results.csv and the appendix also include every noncertifying extremal attempt.

No PD-unsafe states remain unresolved, so additional analytical fallbacks and global viability-grid calculations were unnecessary and were not run.

## Simulator evidence and verification

114/114 extremal directions passed the independent checker. All retained precision attempts reproduced their interval records exactly. Technical tests passed: True; test count: 21. There are 57 bitwise archived-PD replay comparisons and 114 bitwise extremal replay comparisons, including the reused pilot evidence.
Model and Gymnasium first-exit steps agree in 114/114 directions. Maximum observed prefix discrepancy bounds are 1.68602e-15 rad and 5.75473e-15 rad/s. These bounds concern only the saved prefixes and are not implementation-wide error bounds.

Gymnasium uses float32 executed torques, binary floating-point arithmetic, and NumPy sine, whereas the certificate concerns a real-valued rational-parameter map. No justified global implementation-error enclosure is claimed. Simulator agreement does not upgrade the certificate to a proof about all floating-point executions, continuous-time motion, or physical hardware.

## Computational cost and preservation

| Work | Interval updates | Exact generation-repeat updates | Rational transition checks | Generation time (s) | Checker time (s) | Gym steps | Resets |
|---|---:|---:|---:|---:|---:|---:|---:|
| New remaining states | 742 | 742 | 742 | 0.00880912 | 1.05176 | 52000 | 260 |
| Reused pilot | 81 | 81 | 81 | 0.000814584 | 0.0908653 | 5000 | 25 |
| Combined historical total | 823 | 823 | 823 | 0.00962371 | 1.14263 | 57000 | 285 |

Preservation checks passed for 11848 pre-existing files, including Experiments 01–07, the accepted Experiment 08 pilot, D0, and unfinished IQL diagnostics. Source hashes, initial-state identities, exact certificates, rejected/noncertifying attempts, and reproducibility metadata are retained. No previous data were modified; no RL agents were trained; no commit or push was performed.

## Scientific interpretation and limitations

The mathematical certificates answer a different question from the Experiment 07 local solver outcomes: they quantify over every admissible control sequence for the specified discrete model. A failed optimizer alone did not justify this conclusion. The present proof does not rely on solver infeasibility, action-grid failure, or extrapolating an extremal trajectory outside the comparison domain.

The fixed sample was selected through prior PD failure, so these counts are not population-level failure probabilities. The constraints concern raw states at discrete samples, not continuous-time safety between samples. State/reference uncertainty, model mismatch, and an implementation-wide rounding contract are outside the proof. This analysis does not establish an infinite-horizon viable region or a deployable safety filter. Future acquisition can distinguish demonstrated model nonviability from poor learned control, but any changed acquisition domain and recursive safety conditions require separate study.
