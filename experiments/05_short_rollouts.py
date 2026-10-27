"""Experiment 05 technical pilot only; full study requires later approval.

Run: MPLCONFIGDIR=/private/tmp/pendulum_active_rl_mpl python3 \
     experiments/05_short_rollouts.py --pilot

The immutable protocol records future seeds 52--61 but this entry point cannot
launch them. All existing Experiment 01--04 artifacts are protected by hashes.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.horizon_experiment import run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', required=True, help='Run only the approved seed-41 technical pilot.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/rollout_horizon/pilot_seed41')
    parser.add_argument('--resume', action='store_true', help='Resume only with identical source, protocol, data, and environment metadata.')
    args = parser.parse_args()
    run_pilot(ROOT, args.output, args.resume)


if __name__ == '__main__':
    main()
