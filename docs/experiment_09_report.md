# Experiment 09 — Identifying recoverable PD failures

Primary result: **3230/3230 archived PD-unsafe states have independently verified mathematical nonviability certificates**. Verified simulator-safe alternatives found: 0/3230; unresolved model cases: 0/3230.

## Frozen protocol and provenance

The exact Experiment 01 81×81 axes, original masks, binary coordinates and grid IDs were reused. The PD action path preserves the original reconstructed float64 initial observation followed by native float32 Gym observations. All 6,561 baseline episodes were recorded for 200 steps and independently replayed, with safety checked on all 201 raw states. Five accepted pilot baselines and ten pilot interval records were reused unchanged. The primary denominator remains the original 3,230 failures; reference controls never enter it.

The certificate method is unchanged from Experiment 08: omega_next = omega + (15 sin(theta)+3u)/20 and theta_next = theta + omega_next/20, exact rational parameters, exact archived binary initial states, |u|<=2, |theta|<=1, |omega|<=3. On the safe strip the map is monotone; speed clipping is inactive on steps beginning in the safe set. A safe extremal prefix followed by a strict upper crossing under -2 or lower crossing under +2 rules out every safe admissible sequence. Opposite exits never certify. Directed MPFR rounding and the independent exact-rational/Taylor checker validate every premise. Precision escalation remains 128→256→512 only for boundary ambiguity.

## Outcomes and discrepancies

Certificate directions: {'lower': 1615, 'upper': 1615}; components: {'theta': 3230}. Crossing steps: 2–42, times 0.1–2.1 s. Minimum certified crossing separation (decimal display) 2.21447048634e-05. Final precision counts: {'128': 6526}. Exact endpoints and separation lower bounds remain authoritative.

There were 0 nonviability certificates on 32 non-equilibrium PD-safe reference controls. Equilibrium has an exact zero-input witness; other reference states retain empirical PD safety separately from mathematical viability.

Archive/reproduction discrepancy IDs: {'archived_vs_legacy_label_disagreement': [], 'archived_vs_raw_label_disagreement': [], 'legacy_vs_raw_label_disagreement': [], 'archived_vs_legacy_crossing_disagreement': []}. Raw-versus-legacy first-crossing difference IDs: []. Maximum absolute legacy-margin discrepancy: 0. Raw unwrapped margins can differ from archived wrapped-angle minima even when labels and crossing times agree. No archive values were overwritten.

| PD cohort | Episodes | Mean return | Worst return | Mean discounted return | Violating episodes | Minimum raw margin |
|---|---:|---:|---:|---:|---:|---:|
| Whole original grid | 6561 | -155.382357280 | -947.709757405 | -87.044541032 | 3230 (49.230300%) | -5.28318530502 |
| Archived PD-unsafe | 3230 | -314.211208192 | -947.709757405 | -175.466383342 | 3230 (100.000000%) | -5.28318530502 |
| Archived PD-safe | 3331 | -1.369391670 | -7.518567842 | -1.303757284 | 0 (0.000000%) | 0.2 |

All state identities, archived/legacy/raw labels, crossings and margins are in baseline_results.csv. primary_state_results.csv retains all 3,230 certificate outcomes, directions, steps, separations, widths and precisions. direction_results.csv retains every reference and primary extremal attempt, including noncertifying results.

## Gated controller search and optimization

All primary states received valid certificates, so the approved protocol skipped every full-study alternative-controller and optimization evaluation. For each of the eight frozen alternative gain pairs, the number evaluated in the full-study search is zero (not a measured zero success rate). The optimizer subset is empty and full-study scientific solver attempts are zero. No new outcome-selected controller or global viability grid was introduced. The accepted pilot controller/optimization findings remain historical technical evidence and are not extrapolated into full-grid recovery rates.

## Verification and computation

Verification passed: True; protected pre-existing files: 13870. Baseline action replay pairs: 6561; raw-state samples audited: 1318761. Independent certificate archive rechecks: 6526. Every interval generation repeated exactly; source, pilot, case and result hashes are preserved. Technical tests: {'passed': True, 'tests': 3, 'failures': 0, 'errors': 0, 'simulator_steps': 0}. No model/simulator conflict was detected among evaluated trajectories.

New baseline simulator interactions: 2622400 steps, 13112 resets, 36.074850 simulator seconds. Reused pilot baseline evidence corresponds to 2,000 historical steps and ten resets; combined baseline audit evidence is 2,624,400 steps and 13,122 resets. No new extremal Gym rollouts were needed for mathematical certificates. New interval generation/repeat/checker updates: 43582/43582/43582; generator/repeat/checker seconds: 0.446109/0.384284/62.120814. Independent final recheck and saved-array audit costs are recorded separately; no scientific training or full-study solver cost occurred.

## Interpretation and limitations

For these exact 3,230 initial points under the declared real-valued map and constraints, another admissible controller cannot maintain sampled-time safety for 200 steps. This conclusion follows from checked comparison certificates rather than failed search or local optimization. It does not assert universal impossibility for floating-point Gymnasium executions: no implementation-wide error envelope is established. The PD traces agree with the certificate outcomes but do not close that numerical-model gap.

This is a census of the fixed benchmark, not an estimate for all continuous initial conditions. No continuous-time, uncertain-state, model-error, real-world, or infinite-horizon claim is made. The archived PD-safe region is not thereby proved viable or robust. Safe RL research on this benchmark should distinguish infeasible initial states from controller mistakes and investigate a defensible feasible operating domain before seeking safety rescue. Any changed domain, filter, data collection or new experiment requires separate review. Experiments 01–08, the Research Decision Log, all prior results and unfinished IQL files remain unchanged. No commit or push was performed.
