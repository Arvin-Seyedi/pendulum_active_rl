"""B1 read-only verification after authorized research-document updates."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quadrotor_adequacy_finalization import verify
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path,default=Path('/private/tmp/e12_safe_control_gym'));p.add_argument('--full-preservation',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.data_root,ROOT,a.upstream,a.output,a.full_preservation)))
if __name__=='__main__':main()
