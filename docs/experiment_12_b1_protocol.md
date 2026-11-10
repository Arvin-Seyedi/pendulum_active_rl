# B1 archive-only learner adequacy: frozen prospective protocol

User approval is conditional on provenance/preflight before scientific fitting.
The reviewed [design](experiment_12_b1_adequacy_design.md) and
[configuration](../configs/experiment_12_b1_archive_v1.json) define the comparisons.
B0's NO-GO and original FAILED E12/model-agreement gates remain unchanged.

Training IDs in order: 0,4,8,12,5,6,13,14,9,1; nested sizes 4/10. Validation 2,10.
Reused B0 assessment 3,7,11,15 is development only. External H0–H3 are locked
archival supervised assessment, not globally unseen states: their expert/model
behavior was reviewed previously. No external learner predictions before all
twenty prescribed fits, one deterministic repeat and four ridge systems finish.
Entire trajectories stay in one role. No preprocessing is fitted on assessment.

Full 496 inputs preserve B0 feature bytes. Compact ordering: six scaled physical
states; six current-reference values; nonperiodic k/300; two previous grouped
command values; reset flag. Fixed boundary scaling remains unchanged. For this
one known reference, phase reconstructs full preview including terminal padding;
this does not establish sufficiency for varying tasks or MPC optimizer history.
Initial previous-input hover is a sentinel. H archives lack direct motor-submission
streams: command history is source reconstructed from recorded post-interval RPM,
never labeled directly observed execution. Eligibility requires exact reconstruction
of all 4,800 original B0 feature rows, matching request/RPM recurrence, solver-action
targets, recorded current-step indices and native state/reference parameters.
No fallback dataset, epsilon repair or replacement feature is permitted.

Both MLPs have two tanh64 layers and sigmoid2 output (36,098/5,378 parameters).
For seed 123000–123004, SeedSequence.spawn(3) supplies one uint32 Torch seed for
each Linear.reset_parameters. Within representation initial weights match across
sizes; across dimensions hidden/output seeds and weights match, not first-layer
arrays. Minibatch uniforms PCG64(seed+1000000), batch256, floor(u*N), fixed
20,000 Adam1e-3 normalized-MSE updates, checkpoints 5k/20k, no reset/selection.
Adam betas .9/.999, epsilon1e-8, no weight decay, foreach=False, float32 CPU.
Repeat compact/size4/seed123000 through20k; 420,000 total updates. Four float64
ridge systems use alpha.001, unpenalized intercept, mean MSE, original explicit
contractions; retain raw and clipped predictions. No additional fits or tuning.

Primary contrasts: full size10 5k→20k; full→compact at20k/size10; compact
size4→10 at20k. Crossed bootstrap5 seeds×4 whole external starts,20,000 draws,
PCG64-123999: descriptive95% and nominal Bonferroni98.333% intervals. Four
clusters limit power and coverage claims. Optimization/parameterization benefit:
>=25% relative and >=.001 normalized MAE reduction, positive adjusted interval,
no start-mean deterioration>.005; optimization also needs lower training MSE.
Acquisition-design reconsideration separately requires compact20k training
MAE<=.005, external>=.01, <=1.10 compact ridge, >=25% 4→10 reduction and positive
adjusted interval. All criteria are fixed before learner assessment. Report MAE,
MSE, physical units, phase/tail/worst errors and individual trajectories/seeds;
representation/capacity/regularization effects remain confounded. No closed-loop,
safety, acquisition-effectiveness or beyond-expert claim follows.

One worker,4 GiB; zero optimizer/physics operations. Start cumulative
2359.837 CPU/3327.584 active-wall seconds; B1<=600/1200, preflight<=60/120,
task ceilings2959.837/4527.584 within unchanged shared3964.941/6613.063.
Carry1870 MPC calls/294000 steps unchanged. Account each process plus conservative
preparation/reporting. Stop on any mandatory provenance, nonfinite, integrity,
determinism or budget failure; persist partial outputs without retries or tuning.
Accepted sources/results and nine IQL files remain immutable. Generated outputs
are under ignored `results/benchmark_selection/b1_archive_v1/`.

Entry point: pinned Python `experiments/12_b1_archive.py preflight|train|evaluate
--output results/benchmark_selection/b1_archive_v1`. Run stages separately through
the resource wrapper; train rejects an existing fits directory. The source/artifact
seal must pass before training/assessment. Independent export verification is
read-only; it never refits models or substitutes another assessment cohort.
