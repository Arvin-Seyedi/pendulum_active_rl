# E12_B0_ARCHIVE_v1 — frozen supervised gap assessment

Approved archive-only learning, not active acquisition, RL/IQL or policy execution.
Original FAILED E12 qualification and 0.001 model-agreement gates, accepted MPC80
feasibility, Pilot B, authority and nine unfinished IQL files are unchanged.
All new outputs are under fresh `results/benchmark_selection/b0_archive_v1/`,
excluded from Git. Refuse existing fit/assessment outputs; no automatic retries.

## Data and causal information

Use only sixteen accepted original 300-step MPC80 demonstrations (4,800 labels).
Verify repeat identity but exclude all 4,800 repeat labels from fitting/statistical
sample counts. Do not add Pilot B's alternative trajectory. Freeze original paths,
manifest/source hashes, raw reset versus exact requested reset coordinates,
current observation/reference parameters, requested and submitted actions and
runtime identities before fitting. No new expert call or physical transition.

Inputs: current six native physical observations; the known 6×81 task preview
with native terminal padding; k/300; previous executed grouped thrust (hover plus
reset flag at k=0); reset flag. This is 496 scalars. Scale states/reference using
fixed archived constraint midpoints/widths to [-1,1]; scale previous action with
the input bounds. Never estimate normalization on validation/assessment data.
The previous action is one lag, not the current training target. Preview is the
known task reference, not future plant observations. Future actions/states,
rewards, MPC plans, duals, guesses, optimizer memory and private simulator caches
are forbidden features. Full MPC history sufficiency is not established.

## Frozen model and learning curve

The separately versioned B0 uses a deliberately simple feed-forward
496–64–64–2 model, tanh hidden activations, sigmoid normalized outputs. This
replaces the unimplemented prospective GRU *before any learning outcome*, within
the approved instruction to freeze a simple model. One lag cannot reproduce all
teacher/plant memory; residual error may reflect omitted history or approximation
as well as coverage. No architecture search or post-assessment correction.

Training order: `[0,4,8,12,5,6,13,14,9,1]`, with the first four inherited from the
original probe set and the remainder permuted using PCG64 122101. Validation:
`[2,10]`; assessment: `[3,7,11,15]`. No trajectory appears in two partitions.
Learning curves use nested prefixes of 2/4/8/10 trajectories (600/1,200/2,400/3,000
labels). Validation and assessment remain unchanged. All sixteen starts have been
scientifically reviewed previously; assessment is held out from B0 fitting,
not a fresh independent learned-policy evaluation population.

Five seeds 122000–122004. Same initialization and underlying PCG64 minibatch
uniforms per seed across subsets; floor(u*N) maps uniforms to the subset. Adam
1e-3, betas .9/.999, eps 1e-8, no weight decay, foreach=False; batch 256, normalized
MSE, float32, CPU one thread, deterministic algorithms, exactly 5,000 updates.
Uniform transition batches preserve whole-trajectory partitioning; equal-length
trajectories have equal training weight. Log train loss/gradient norms every 500
updates; use only the last checkpoint, no validation-based stopping or selection.
Train 20 fits, plus a prespecified independent replay of seed122000/size4. Require
bitwise initial/final parameters, optimizer moments, schedules/history and restored
predictions. Validation is diagnostic only. Do not unlock assessment until all
fits and the deterministic repeat pass.

Simple comparator: ridge regression on the identical 496 inputs, mean normalized
MSE plus 1e-3 L2 coefficient penalty, unpenalized intercept, float64 solve with
backward-error check <=1e-12. Report raw predictions/clipping separately; clipped
[0,1] predictions are the preregistered primary baseline. Fit once per subset;
do not count deterministic ridge repeats as seed replication.
Matrix products use explicit NumPy contractions. Before sealing, the pinned
matmul path emitted floating-status warnings on a finite synthetic fixture;
its predictions matched an independent contraction exactly and its backward
error was 1.98e-16. That diagnostic is retained; no warning is suppressed, runtime
version substituted, objective changed or regularization tuned.

## Metrics and prospective decision

Report train/validation/assessment errors separately, each with whole-trajectory
macro MAE/RMSE, per-channel physical N and normalized units, per-trajectory p95/max,
worst trajectory and early (0–49), middle (50–199), late (200–299) phases.
Coverage distances condition on reference step and use fixed state ranges;
distance/error correlation is descriptive with no transition-level p-value.
Report near-bound behavior and early versus late support. Supervised metrics do
not measure closed-loop return, safety, robust control or acquisition efficiency.

The primary comparison is size10 minus size4, paired by seed and assessment
trajectory. Freeze 20,000 crossed seed/trajectory bootstrap draws, PCG64 122999,
percentile95 intervals; never bootstrap adjacent transitions as independent units.
Four assessment starts and five optimizer seeds permit only conditional, limited
uncertainty statements. Intermediate sizes are descriptive, not model-selection
candidates and not required to be monotone.

GO only for *designing* future acquisition if all of the following hold:

- Base4 assessment macro normalized MAE >=0.01, with bootstrap lower bound >=0.005.
- Size10 assessment MAE <=75% of base4; paired improvement CI lower bound >0.
- No assessment trajectory's seed-mean MAE worsens by >0.005; early-phase mean
  error does not increase.
- Size10 MLP error <=1.10 times ridge error, avoiding a clear weak-model artifact.
- All provenance/preflight/finite-value/checkpoint/reproducibility checks pass.

Here 0.01 equals about 0.002404 N (1% of the actual channel range), one fifth of
the frozen 5%-range action perturbation; 0.005 excludes a negligible half-scale
gap, while 25% reduction is the original proposed supervised learning criterion.
These are operational supervised thresholds, not derived physical safety bounds
or evidence of a particular closed-loop sensitivity. A failed or uncertain gate
means NO-GO for acquisition now; it does not prove another architecture/data source
cannot help. The original 10% reward-headroom criterion remains unestablished.

## Resource and preservation contract

Carry cumulative 2,164.941 CPU /3,013.063 active-wall seconds. This B0 approval adds
at most 1,800 CPU /3,600 active-wall seconds: new ceilings **3,964.941 /6,613.063**.
Do not add unused earlier allowance again. Include preflight, fitting, baseline,
evaluation, deterministic repeat, persistence and preservation work. One worker,
4 GiB, no GPU; OS CPU/remaining-wall limits plus periodic checks. Stop on identity,
leakage, nonfinite arithmetic, reproducibility failure or resource exhaustion.
Preserve partial failures and do not quietly reduce schedules or tune settings.
Original 1,870 MPC solves /294,000 physics steps remain unchanged, with zero B0
use. No old ledger or accepted scientific artifact is rewritten. No commits/pushes.
