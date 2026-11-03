"""Independent rational checks of E11B evidence; no MPFR or plant evaluation.

This checker does not import the producer's interval arithmetic or residual
functions. Interval endpoints are interpreted as exact integer ratios.
"""
from fractions import Fraction as F
from pathlib import Path
import gc
import json
import hashlib
import subprocess
import time
import numpy as np
from src.drilling_reproduction import read,write,sha
from src.drilling_archive_verification import check_hashes,check_preservation

CONTEXT_FILES={'CURRENT_STATUS.md','RESEARCH_CONTEXT.md','docs/RESEARCH_DECISION_LOG.md'}
PILOT_CONTEXT_COMMIT='f558a13b3eab01d9bc7ad1b628cc72d751958aa9'


def verification_destination(path,archive):
    """Finalization outputs must be new files outside the accepted evidence."""
    path,archive=Path(path).resolve(),Path(archive).resolve()
    if path.is_relative_to(archive):raise ValueError('Cannot write into accepted evidence')
    if path.exists():raise FileExistsError(path)
    return path


def preserved_project_file(root,name,expected):
    """Only reviewed context documents may differ from the pilot snapshot."""
    if sha(Path(root)/name)==expected:return
    if name not in CONTEXT_FILES:raise AssertionError('Preserved file changed: '+name)
    historical=subprocess.check_output(['git','show',PILOT_CONTEXT_COMMIT+':'+name],cwd=root)
    if hashlib.sha256(historical).hexdigest()!=expected:
        raise AssertionError('Historical context identity mismatch: '+name)


def rational(pair):return F(int(pair[0]),int(pair[1]))
def dyadic(x):return F.from_float(float(x))
def endpoints(record):return rational(record['lo']),rational(record['hi'])


def dot_bounds(row,x):
    lo=hi=F(0)
    for record,v in zip(row,x):
        a,b=endpoints(record);v=dyadic(v)
        lo+=min(a*v,b*v);hi+=max(a*v,b*v)
    return lo,hi


def check_worker_manifest(out):
    """Check worker evidence and the preserved pre-resume supervisor record.

    The worker finished before the supervisor wrote its final execution status.
    On this resumed audit its manifest includes the initial status at a mutable
    path. Verify those exact bytes at their preserved snapshot, then independently
    check the final supervisor status. No scientific artifact hash is excepted.
    """
    out=Path(out);hashes=dict(read(out/'completion.json')['artifact_sha256'])
    relinked=False
    if 'execution_limits.json' in hashes and sha(out/'execution_limits.json')!=hashes['execution_limits.json']:
        initial=out/'execution_limits_initial.json'
        assert initial.exists() and sha(initial)==hashes['execution_limits.json'], 'Unexplained execution metadata change'
        hashes['execution_limits_initial.json']=hashes.pop('execution_limits.json')
        relinked=True
    check_hashes(out,hashes)
    return relinked


def verify(root,out,full_preservation=True,report_path=None,code_root=None):
    started=time.perf_counter();root,out=Path(root),Path(out)
    code=Path(code_root or root)
    destination=verification_destination(report_path,out) if report_path is not None else out/'independent_verification.json'
    cfg=read(out/'protocol.json');relinked=check_worker_manifest(out)
    assert read(out/'execution_limits.json')['returncode']==0
    a=dict(np.load(out/'frozen_binary64.npz'));exact=read(out/'exact_dyadic.json')
    # Independent loop ordering from serialized dyadic operands.
    residuals=[];kvs=[]
    for i in range(10):
        rr=dyadic(a['F_M_d'][i,0]);kk=F(0)
        for j in reversed(range(10)):
            rr-=dyadic(a['D_M'][i,j])*dyadic(a['Vs'][j,0])
            rr-=dyadic(a['K_M'][i,j])*dyadic(a['IC'][j,0])
            kk+=dyadic(a['K_M'][i,j])*dyadic(a['Vs'][j,0])
        residuals.append(rr);kvs.append(kk)
    assert residuals==[rational(x) for x in exact['r0_exact']]
    assert kvs==[rational(x) for x in exact['KM_Vs_exact']]
    assert any(residuals) and any(kvs)
    high=[read(out/f'interval_{p}.json') for p in (128,256)]
    overlap=0
    for h in high:
        m=h['matrix_intervals']
        for i in range(10):
            kl,kh=dot_bounds(m['KM'][i],a['Vs'][:,0])
            dl,dh=dot_bounds(m['DM'][i],a['Vs'][:,0])
            il,ih=dot_bounds(m['KM'][i],a['IC'][:,0])
            fl,fh=endpoints(m['FM'][i][0]);rlo,rhi=endpoints(h['r0'][i]);vlo,vhi=endpoints(h['KM_Vs'][i])
            assert rlo<=fl-dh-ih<=fh-dl-il<=rhi
            assert vlo<=kl<=kh<=vhi
    for key in ('r0','KM_Vs'):
        for low,high_record in zip(high[0][key],high[1][key]):
            al,ah=endpoints(low);bl,bh=endpoints(high_record)
            assert max(al,bl)<=min(ah,bh),'Disjoint precision enclosures'
            overlap+=1
    point_checks=0
    rounding_checks=0
    point_rows=read(out/'pointwise_equations.json')
    unit_roundoff=F(1,2**53);gamma=100*unit_roundoff/(1-100*unit_roundoff)
    for case in cfg['cases']:
        with np.load(out/f'{case}_original_points.npz') as x,np.load(out/f'{case}_repeat_points.npz') as y:
            assert all(x[k].tobytes()==y[k].tobytes() for k in x.files)
            for j,step in enumerate(cfg['indices']):
                e=list(map(dyadic,x['E'][:,j]));t=dyadic(x['t'][0,j])
                force=np.linalg.solve(a['M1_global'],x['F'][:,j])[::2]
                # Direct physical equation versus native error equation,
                # cancelling the common recorded force before evaluation.
                for i in range(10):
                    value=dyadic(a['F_M_d'][i,0])
                    for k in range(10):
                        q=e[k]+dyadic(a['IC'][k,0])+t*dyadic(a['Vs'][k,0])
                        vel=e[10+k]+dyadic(a['Vs'][k,0])
                        value+=dyadic(a['K_M'][i,k])*(e[k]-q)
                        value+=dyadic(a['D_M'][i,k])*(e[10+k]-vel)
                    assert value==residuals[i]-t*kvs[i]
                    magnitude=2*abs(dyadic(force[i]))+abs(dyadic(a['F_M_d'][i,0]))
                    for k in range(10):
                        magnitude+=abs(dyadic(a['K_M'][i,k]))*(abs(e[k])+abs(dyadic(a['IC'][k,0]))+abs(t*dyadic(a['Vs'][k,0])))
                        magnitude+=abs(dyadic(a['D_M'][i,k]))*(abs(e[10+k])+abs(dyadic(a['Vs'][k,0])))
                    observed=dyadic(point_rows[case+'_original'][j]['equation_difference'][i])
                    assert abs(observed-value)<=gamma*magnitude,'Exact rational rounding bound exceeded'
                    rounding_checks+=1
                point_checks+=1
    for name,h in read(out/'preservation_before.json').items():preserved_project_file(root,name,h)
    source=read(out/'source_identities.json')
    amendments=read(out/'implementation_amendments.json')['files'] if (out/'implementation_amendments.json').exists() else {}
    maintenance=read(out/'postrun_source_maintenance.json')['files'] if (out/'postrun_source_maintenance.json').exists() else {}
    for name,h in source['implementation_sha256'].items():
        expected=amendments.get(name,{}).get('current_sha256',h)
        if name in maintenance:
            assert maintenance[name]['run_sha256']==expected,name
            assert sha(out/'implementation_snapshot_completed'/name)==expected,name
            expected=maintenance[name]['current_sha256']
        assert sha(code/name)==expected,name
        if name in amendments:assert sha(out/'implementation_snapshot_initial'/name)==h,name
    old=root/cfg['accepted_evidence']
    if full_preservation:
        check_preservation(root,read(old/'preservation_before.json')['sha256'])
    for name,h in read(old/'authority_preservation_before.json').items():assert sha(name)==h,name
    execution=read(out/'execution_limits.json')
    assert execution['peak_sampled_rss_bytes']<=cfg['rss_bytes_cap']
    assert read(out/'summary.json')['cpu_seconds']<cfg['cpu_seconds_cap']
    result={'passed':True,'rational_residual_vectors_recomputed':2,
        'interval_residual_components_checked':40,'cross_precision_components_overlap':overlap,
        'independent_original_point_equations':point_checks,'archived_repeat_pairs_bitwise':6,
        'exact_rational_componentwise_rounding_bounds_checked':rounding_checks,
        'all_prior_project_files_checked':len(read(old/'preservation_before.json')['sha256']) if full_preservation else None,
        'initial_supervisor_metadata_verified_at_preserved_snapshot':relinked,
        'final_supervisor_metadata_sha256':sha(out/'execution_limits.json'),
        'new_trajectories':0,'producer_MPFR_code_imported':False,'seconds':time.perf_counter()-started}
    write(destination,result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    verify(root,args.output or root/read(root/'configs/experiment_11b.json')['output'])
