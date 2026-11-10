# Experiment 12 B1 — archive-only learner adequacy

**Completed; acquisition-design NO-GO under the prospectively frozen B1 criteria.** B0’s historical NO-GO and the original FAILED E12 overall /0.001 model-agreement gates remain unchanged. No learned policy was deployed; no MPC solve, physics step, acquisition or RL/IQL experiment was executed.

## Provenance and prospective controls

All 4,800 original B0 float32 feature rows reproduce **bitwise** from the recorded RPM/native command recurrence. H0–H3 provide 1,200 requested-action labels from four complete 300-step trajectories; labels equal saved solver first actions, states equal solver state parameters, reference previews and recorded control indices align exactly. Native request→clipping→RPM conversion matches all records. The 160 scripted native boundaries are mocks, not simulations.

H archives **do not contain direct motor-submission streams**. Their previous-command feature is explicitly source-reconstructed from post-interval recorded RPM, cross-checked against recorded requests; it is not an observed historical actuator measurement. The same reconstruction reproduces B0’s directly submitted command features exactly. Times are causal control indices/50, not measured wall timestamps. Reset hover is a sentinel. H0–H3 were previously reviewed for expert/model diagnostics, never used in B0 fitting, and have no exact initial-coordinate duplicates with B0; they are not globally unseen or a fresh closed-loop cohort. All four were sealed before fitting, with no outcome-dependent exclusions.

Training order 0,4,8,12,5,6,13,14,9,1; nested 4/10; validation 2,10. Old assessment 3,7,11,15 is reused **development-only**. All prescribed fits finished before external predictions. Fixed constraint scaling is unchanged; no assessment-fitted preprocessing.

Full 496 and compact 16 representations use physical state 6, current reference 6, nonperiodic phase 1, previous command 2 and reset 1; full input additionally carries the81-step preview. For this fixed known reference, phase reconstructs preview/padding. Missing optimizer, plan/dual, warm-start and simulator history remain an identifiability limitation; neither representation is proven Markov-sufficient.

MLPs: tanh 64/tanh 64/sigmoid 2, 36,098 versus 5,378 parameters; seeds 123000–123004; normalized MSE; Adam 1e−3, betas(.9,.999), epsilon 1e−8, batch 256, no decay, foreach=False, float32 single-thread CPU. Layer-seeded initialization pairs hidden/output streams across representations and all weights within representation across sizes. PCG64(seed+1000000) minibatch uniforms are paired. Twenty fits continue unchanged to 20k, retaining 5k/20k; compact4/seed123000 repeats 20k:420,000 total updates. Four ridge fits retain mean-MSE+alpha .001, unpenalized intercept, float64 and explicit contractions, raw/clipped predictions. No tuning, early stopping or new arms.

## Supervised results

MAE is macro over whole trajectories and action channels, then five MLP seeds. A normalized unit is the action width 0.24035951867699623 N. All values below are supervised errors, not control returns or safety metrics.

| Representation | Train trajectories | Updates | Train MAE | Validation MAE | External MAE | External MAE (mN) | External MSE (normalized²) | Worst action error (N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 4 | 5000 | 0.00993554 | 0.01677383 | 0.02132967 | 5.126790 | 0.004045533 | 0.231338973 |
| full | 4 | 20000 | 0.00533127 | 0.01560715 | 0.01585344 | 3.810525 | 0.003297179 | 0.239488356 |
| full | 10 | 5000 | 0.01341163 | 0.01758038 | 0.01683522 | 4.046507 | 0.002189585 | 0.233579005 |
| full | 10 | 20000 | 0.00876579 | 0.01558383 | 0.01246805 | 2.996815 | 0.002243959 | 0.239615400 |
| compact | 4 | 5000 | 0.00770385 | 0.01807859 | 0.02131506 | 5.123277 | 0.005538757 | 0.233881812 |
| compact | 4 | 20000 | 0.00545346 | 0.02277534 | 0.02233899 | 5.369389 | 0.007525978 | 0.239114351 |
| compact | 10 | 5000 | 0.01001668 | 0.01250484 | 0.01383579 | 3.325564 | 0.002342293 | 0.229580890 |
| compact | 10 | 20000 | 0.00606367 | 0.00967667 | 0.00967539 | 2.325571 | 0.001570544 | 0.207407363 |

The retained JSON contains training/validation/external MAE and MSE in normalized and physical N/N² units, RMSE, p95, near-bound errors, all seed/trajectory/phase records, worst-action indices and reused-development diagnostics. Equal updates control sampled exposures, not parameter count or equal FLOPs.

| Ridge representation / size | Train MAE | Validation MAE | External MAE | External MAE (mN) | External MSE |
|---|---:|---:|---:|---:|---:|
| full_n4 | 0.00835586 | 0.01020699 | 0.01289733 | 3.099995 | 0.004098729 |
| full_n10 | 0.01006650 | 0.01085803 | 0.01305092 | 3.136912 | 0.003916817 |
| compact_n4 | 0.00940472 | 0.01112498 | 0.01378284 | 3.312836 | 0.004097040 |
| compact_n10 | 0.01108888 | 0.01197108 | 0.01398964 | 3.362544 | 0.003929234 |

## Prespecified paired contrasts

Positive differences mean error reduction. Crossed bootstrap resamples five seeds and four whole external starts, 20,000 draws PCG64-123999. Intervals are descriptive 95% and nominal Bonferroni-adjusted 98.333%; four clusters do not support guaranteed coverage or broad population claims. No transition bootstrap.

| Contrast | MAE reduction | Relative reduction | 95% CI | 98.333% CI | Seed-mean sample SD | Seed-mean range | Practical benefit gate |
|---|---:|---:|---|---|---:|---|---|
| optimization | 0.00436717 | 25.94% | [0.00205377, 0.00646568] | [0.00149565, 0.00684909] | 0.00269023 | [0.00003767, 0.00690847] | True |
| parameterization | 0.00279267 | 22.40% | [0.00064899, 0.00516861] | [0.00019012, 0.00574909] | 0.00219186 | [0.00129347, 0.00667313] | False |
| data | 0.01266360 | 56.69% | [0.00733605, 0.01632740] | [0.00673121, 0.01684925] | 0.00103791 | [0.01108802, 0.01399174] | data contrast only |

Optimization passes the fixed >=25%,>=.001 MAE and adjusted-positive-CI gates, with lower full-training MSE and no start-mean deterioration>.005. However full size10 external MSE **increases** from 0.002189585 to 0.002243959 (~2.48%); lower average absolute error is not uniform error improvement. The worst action error also increases.

Parameterization’s 22.40% MAE reduction has a positive adjusted interval but **fails 25%**, without changing the threshold. Size10 compact20k beats both size10 ridges on this external cohort. This practical representation contrast also changes parameter count, optimization geometry and effective regularization; it is not a pure capacity-controlled causal result.

Compact4→10 improves MAE 56.69% with a positive adjusted interval; every start’s five-seed mean improves. Nevertheless compact4 training longer reduces training loss while external MAE worsens 0.02131506→0.02233899 and external MSE rises 0.00553876→0.00752598. This supports a measured generalization/metric limitation at low data, not proof of data scarcity or a useful acquisition ranking.

## Phase, coverage and tails

| Compact20k / size | Early MAE (0–1s) | Middle MAE (1–4s) | Late MAE (4–6s) | Same-phase nearest-context distance: early / middle / late |
|---|---:|---:|---:|---|
| 4 | 0.09872224 | 0.00921711 | 0.00383019 | 0.10391958 / 0.0037759519 / 3.7321974e-06 |
| 10 | 0.03681054 | 0.00459706 | 0.00372531 | 0.044181863 / 0.0011133274 / 1.0294086e-06 |

Distance is RMS normalized16-input distance to a training context at exactly the same phase, not a physical feasible-set distance. Extra training starts reduce this distance most visibly in transients; late inputs are already highly redundant. Size10 descriptive phase-conditioned distance/error Spearman correlations are 0.5060/0.7237/−0.6508. Correlated temporal samples, phase and fitting effects preclude independent-step significance or acquisition causality. The earlier training-only conditioning audit remains unchanged (full preview redundancy/rank and ~96% summed variance); compactness does not itself resolve hidden history.

Compact size10/20k external start-mean MAEs H0–H3: 0.00857249, 0.01192392, 0.00595997, 0.01224516. Its worst individual action error is 0.207407363 N (~86.3% of range); full size10/20k reaches 0.239615400 N (~99.7%). These rare large errors prohibit interpreting small average imitation error as safety.

## Frozen decision and limitations

- `training_MAE_at_most_.005`: **False**.
- `external_MAE_at_least_.01`: **False**.
- `external_MAE_within_1.10_ridge`: **True**.
- `data_relative_reduction_at_least_.25`: **True**.
- `data_adjusted_CI_positive`: **True**.

**NO-GO for designing new targeted-versus-random acquisition under B1’s frozen adequacy rule.** Compact 10/20k training MAE 0.00606367 exceeds 0.005; external MAE 0.00967539 is below the required 0.01 practical gap. Its ridge comparison and data-contrast gates pass, but all gates are required. The result is not evidence that informative acquisition is impossible. No threshold relaxation, replacement model/cohort, new seed or extra fitting is justified automatically. B0’s 9.53% result/25% gate stays unchanged; B1 uses new prescribed seeds and a different archival external cohort.

Optimization contributes to the B0 concern, but cannot alone explain all behavior. Representation/capacity/regularization, transient coverage, hidden teacher history and optimizer allocation remain confounded. Five seeds, four previously expert-reviewed archival starts, one trajectory subset order and one known reference limit generalization. Teacher-forced past commands may differ under autonomous rollout. No return, safety, controller-feasibility or beyond-expert conclusion follows. Any future acquisition direction requires separate review and independent closed-loop evaluation, not reuse of this assessment as untouched evidence.

## Verification, preservation and resources

Phase A completed before fitting: exact B0 reconstruction, H provenance/causality, immutable roles/sources, full prior-preservation audit and 211 zero-solve regression tests. A first preflight output-path rejection was preserved; only output routing changed to respect the original B0 verifier guard, with no relaxed scientific criterion. The full B1 source/configuration seal is stored in `protocol_seal.json`.

Independent workspace verification passes all 40 checkpoints (independent binary64 versus saved float32 maximum prediction difference 1.04775e−6), all four ridge systems (maximum backward error 1.03297e−16), fixed schedules/initialization pairing, three adjusted CIs and bitwise repeat weights/Adam moments/histories at both checkpoints. Preservation checks cover 263 tracked originals, 12 preexisting untracked files, all 204 B0 artifacts, 87 authoritative files, 87,315 accepted MPC80 files, 59,268 prior result/data entries and 12,147 Pilot B entries (overlapping historical scopes, not summed as unique). Nine untracked IQL files remain unchanged and unimported. Clean-export verification and regression outcomes are recorded separately.

Generated weights, processed archival arrays, predictions and evidence remain outside Git under `results/benchmark_selection/b1_archive_v1/`. See [protocol](experiment_12_b1_protocol.md) and [read-only reproduction](experiment_12_b1_reproduction.md). Resource totals and clean-export completion are finalized below without rerunning scientific fits.

Clean-export independent reconstruction **PASSED**; workspace and clean-export regression each **211/211 PASS**, zero failures/errors/skips. No extra scientific fit was used for verification. Peak recorded scientific-fit RSS was 461,160,448 bytes, below4 GiB; one worker and single-thread CPU were used.

B1 consumed **248.278 CPU /321.828 active-wall seconds**, including a conservative final reporting/metadata charge; approved maxima600/1200. Cumulative charges are **2608.115/3649.412**. Remaining shared time is **1356.826 CPU /2963.651 wall seconds**, within unchanged shared ceilings3964.941/6613.063. Unused B1-local allowance is351.722/878.172; it is not another authorization. **1,870 MPC calls /294,000 physics steps remain unchanged**; B1 used zero. No tracked historical file, context document, decision-log entry or accepted result was changed; nothing was committed or pushed.
