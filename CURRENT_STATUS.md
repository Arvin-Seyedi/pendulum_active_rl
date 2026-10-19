# Safe RL Research — Current Status

## Current stage

The project is in the Pendulum-v1 active-query proof-of-concept stage.

Immediate research goal:
test whether a learner can identify states where its current offline dataset is insufficient and request targeted demonstrations from an existing controller.

## Completed

### Pendulum expert
PD controller:
`u = -8*theta - 2*omega`

Torque limit:
`[-2, 2] N*m`

Safety:
- `|theta| <= 1 rad`
- `|omega| <= 3 rad/s`

### Expert safe/unsafe map
Grid:
- `81 x 81`
- `6561` initial conditions
- `theta_0 in [-0.8, 0.8]`
- `omega_0 in [-2, 2]`
- 10 s rollouts

Verified:
- Safe: `3331`
- Unsafe: `3230`
- Safe percentage: `50.77%`
- Earliest violation: `0.10 s`
- Mean first violation: `0.4939 s`
- Latest first violation: `2.10 s`

### Initial offline dataset D0
Nominal region:
- `theta_0 in [-0.25, 0.25]`
- `omega_0 in [-0.5, 0.5]`

Dataset:
- `100` trajectories
- `20,000` transitions
- `0` safety violations
- minimum safety margin: `0.7348`

### Repository
Repository:
`pendulum_active_rl`

Current structure:

```text
pendulum_active_rl/
├── data/
├── experiments/
│   ├── 01_expert_map.py
│   ├── 02_generate_D0.py
│   └── 03_uncertainty_map.py
├── results/
├── src/
│   ├── active_query.py
│   ├── dataset.py
│   ├── ensemble.py
│   ├── expert.py
│   └── safety.py
├── .gitignore
├── README.md
└── requirements.txt
```

Implemented and verified:
- `src/expert.py`
- `src/safety.py`
- `src/dataset.py`
- `experiments/01_expert_map.py`

`01_expert_map.py` reproduces the verified `50.77%` safe result.

## Immediate next step

Implement:

`experiments/02_generate_D0.py`

inside the structured repository so it regenerates the verified 100-trajectory / 20,000-transition D0 dataset.

After that:
- implement `src/ensemble.py`
- implement `experiments/03_uncertainty_map.py`
- train a behavior-cloning ensemble on D0
- map `U(s) = Std[pi_1(s), ..., pi_N(s)]`

Desired first result:
- low uncertainty in the D0-covered region
- higher uncertainty outside dataset support

## Important methodology

Uncertainty:
- policy-ensemble disagreement

Safety criticality:
- known safety margin

Future query priority should consider:
- uncertainty
- safety criticality
- optionally distance from dataset support

Later comparison must use equal additional data budgets:

`D0 + targeted expert data`
versus
`D0 + random expert data`

## After account migration

1. Read `RESEARCH_CONTEXT.md`.
2. Read this file.
3. Inspect the current repository before changing code.
4. Re-run `experiments/01_expert_map.py` if environment verification is needed.
5. Do not restart the project from scratch.
