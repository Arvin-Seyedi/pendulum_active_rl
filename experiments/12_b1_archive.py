"""Separately versioned B1; preflight must precede training and assessment."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import quadrotor_adequacy_experiment as e
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['preflight','train','evaluate']);p.add_argument('--data-root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path,default=Path('/private/tmp/e12_safe_control_gym'));a=p.parse_args()
    result=getattr(e,a.stage)(a.data_root,a.upstream,a.output)
    print(json.dumps({'stage':a.stage,'complete':True,'decision':result.get('decision') if isinstance(result,dict) else None}))
if __name__=='__main__':main()
