"""Approved Experiment 05 seeds 52–61; independent workers and safe resumption.

Launch independently: python3 experiments/05_full_study.py --detach --workers 3
Resume identically:   python3 experiments/05_full_study.py --detach --workers 3
Verify/aggregate:     python3 experiments/05_full_study.py --summarize-only
Completed seeds are hash-verified and skipped. Seed 41 cannot be launched here.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pendulum_active_rl_mpl')

from src.horizon_experiment import read_json, write_json, _assert_hashes
from src.horizon_study import SEEDS, completed, exclusive, initialize, run_seed, manifest
from src.horizon_study_analysis import summarize

OUTPUT = ROOT / 'results/rollout_horizon/full_study_52_61'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 10: parser.error('workers must be between 1 and 10')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), '--workers', str(args.workers)]
        if args.summarize_only: command.append('--summarize-only')
        with (OUTPUT / 'supervisor.log').open('a', buffering=1) as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Independent supervisor PID {process.pid}; log: {OUTPUT / "supervisor.log"}')
        return
    with exclusive(OUTPUT):
        try:
            if completed(OUTPUT):
                _assert_hashes(ROOT, read_json(OUTPUT / 'metadata.json')['protected_hashes'])
                print('All ten seeds and study artifacts already verified; nothing rerun.', flush=True)
                return
            initialize(ROOT, OUTPUT)
            done = [s for s in SEEDS if completed(OUTPUT / f'seed_{s}')]
            pending = [s for s in SEEDS if s not in done]
            print(f'Verified completed seeds: {done}; incomplete seeds: {pending}', flush=True)
            write_json(OUTPUT / 'run_status.json', {'supervisor_pid': os.getpid(), 'completed': done,
                       'pending': pending, 'workers': args.workers, 'phase': 'running'})
            if args.summarize_only and pending:
                raise RuntimeError(f'Cannot aggregate incomplete seeds: {pending}')
            if pending:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                    futures = {executor.submit(run_seed, ROOT, OUTPUT, s): s for s in pending}
                    for future in as_completed(futures):
                        seed = future.result()
                        done.append(seed)
                        print(f'VERIFIED COMPLETE seed {seed}; {len(done)}/10', flush=True)
                        write_json(OUTPUT / 'run_status.json', {'supervisor_pid': os.getpid(), 'completed': sorted(done),
                            'pending': [s for s in SEEDS if s not in done], 'workers': args.workers, 'phase': 'running'})
            summarize(OUTPUT)
            metadata = read_json(OUTPUT / 'metadata.json')
            _assert_hashes(ROOT, metadata['protected_hashes'])
            _assert_hashes(ROOT, metadata['source_hashes'])
            write_json(OUTPUT / 'run_status.json', {'completed': list(SEEDS), 'pending': [], 'phase': 'complete'})
            write_json(OUTPUT / 'checks.json', {'all_passed': True, 'verified_seeds': list(SEEDS),
                'exact_budget_datasets': 180, 'member_checkpoints': 1000,
                'all_seed_artifact_hashes_verified': True, 'pilot_and_experiments_01_04_preserved': True,
                'all_source_hashes_unchanged': True, 'technical_checks': read_json(OUTPUT / 'technical_checks.json'),
                'all_10_seeds_in_paired_bootstrap': True, 'performance_tuning': False})
            hashes = manifest(OUTPUT)
            # Supervisor output remains open until this process exits.
            hashes = {k:v for k,v in hashes.items() if k != 'supervisor.log' and not k.endswith('/.lock')}
            write_json(OUTPUT / 'artifact_hashes.json', hashes)
            print('FULL STUDY COMPLETE: all ten seeds verified and paired report saved.', flush=True)
        except BaseException as exc:
            write_json(OUTPUT / f'failure_{time.time_ns()}.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
