"""Read-only E12_MMAQ record verification; no execution or screening mode."""
from pathlib import Path
import argparse
import os
import sys
import time
import resource
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib';os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import quadrotor_pilot as p
from src.quadrotor_mmaq_record_audit_v1 import audit_all_records
from src.quadrotor_mmaq_v1 import safe_write


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    try:
        result,partial=audit_all_records(ROOT,a.upstream,a.output)
        print({'passed':result['passed'],'records':result['records'],'partial_records':len(partial['partial_records']),
               'unmeasured_failed_calls':partial['unmeasured_failed_calls']})
    finally:
        usage=p.read(a.output/'usage.json');usage['cpu_seconds']+=time.process_time()
        safe_write(a.output/'usage.json',usage)
        safe_write(a.output/'postrun_audit_resources.json',{'cpu_seconds':time.process_time(),
                   'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':main()
