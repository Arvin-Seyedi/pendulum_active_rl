"""Read-only verification of completed E11A records; no MATLAB launch or simulation."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT,help='Root holding the accepted archive and authority inputs')
    p.add_argument('--output',type=Path,help='Optional new JSON file outside the accepted archive')
    a=p.parse_args()
    from src.drilling_archive_verification import outside_archive,verify_archive
    if a.output:
        outside_archive(a.output,a.project_root/'results/drilling_reproduction/native_pilot')
        if a.output.exists():p.error('Refusing to overwrite verification evidence')
    result=verify_archive(a.project_root,ROOT)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','native_reference_residuals']},indent=2))
