"""E11A frozen provenance and native execution; no controller implementation."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile

PREFIX='8_Downhole_Directional_Drilling_Barrier_Avoidance/Simulation/'
MAIN='Direct_Drilling_5_Node_lateral_constrainted_LPV_barrier.m'
DEPENDENCIES=['IC.mat','Vs.mat','force_d.mat','torque_d.mat','WOB_d.mat','TOB_d.mat',
              'Mat_data2.mat','err.mat','K_vl.m','K_vu.m']
EXPECTED={MAIN:'4b51369ccaeb8f016807b91f6bc4d4b2e93c20a13d6aef24b2ed48dba6ada3a0',
 'Chart 4 ICs/'+MAIN:'9b4162472db0f53833fbdbbaba95da892736d9a032fe322dc8be211a703875a4',
 'Chart 4 ICs/IC_Test.m':'3b2679f7cdfe29cbabc7536ab69077d118c937d02dddd589eff08c081eeda55c',
 'Chart 4 ICs/Chart_Plot.m':'5e30b08fbda890c4087c292e43d74b9065b99fdf8eee7610ec89cfe23c9cffa5',
 'Mat_data2.mat':'7bba294909e3eaac7816e87fe5c7ff095aa6631d4828853e74d6413694a05093'}
A='-0.0005,-0.1,0.0005,0.1,0.0005,0.1,0.0001,0.1,0.0001,0.1,0.00001,0,-0.00001,-0,0.00001,0,0,-0'
B='-0.0005,-0.5,0.0005,0.5,0.0005,0.5,0.0001,0.1,0.0001,0.1,0.00001,0.1,-0.00001,-0.1,0.00001,0.1,0,-0'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def immutable(path,value):
    if Path(path).exists(): assert read(path)==value,str(path)
    else: write(path,value)

def cases():
    values=[f'[{A},-0.00045363,-1.7321]’',f'[{B},0.00018,-3.6136]’',
            f'[{B},-0.00045363,-1.7321]’', '[zeros(1,18),0.00056292,4.5930]’',
            f'[{B},0.0029012,2.1651]’','zeros(20,1)']
    return [dict(case_id=f'P{i}',initial_expression=v.replace('’',"'"),
            kernel=MAIN if i==0 else 'Chart 4 ICs/IC_Test.m',
            definition=f'{MAIN}:326-327' if i==0 else
                ('Newly approved zero-error technical input; not an archived case' if i==5 else
                 'Chart 4 ICs/Chart_Plot.m:ic'+str(i)),
            archived_case=i!=5) for i,v in enumerate(values)]

def freeze(root):
    root=Path(root).resolve();config=read(root/'configs/experiment_11a.json');out=root/config['output']
    archive=(root/config['archive']).resolve();assert sha(archive)==config['archive_sha256']
    assert config['workers']==1 and config['states']==200001 and config['transitions']==200000
    assert not any(config[k] for k in ('source_modification','integration_refinement','actuator_saturation',
                                      'controller_synthesis','alternative_actions','training'))
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        hashes={n.removeprefix(PREFIX):hashlib.sha256(z.read(n)).hexdigest()
                for n in z.namelist() if n.startswith(PREFIX) and not n.endswith('/')}
        for name,h in EXPECTED.items():assert hashes[name]==h,name
        for folder in ['Chart 4 ICs/','Copy_of_Chart 4 ICs/','Surface Plot/']:
            for name in DEPENDENCIES:assert hashes[folder+name]==hashes[name],folder+name
    immutable(out/'protocol.json',config)
    immutable(out/'case_manifest.json',{'cases':cases(),'archive':str(archive),'members_sha256':hashes,
        'standalone_Chart_script_is_not_IC3':True,'initial_vectors_evaluated_by_native_MATLAB':True})
    sources=['configs/experiment_11a.json','src/drilling_reproduction.py','src/drilling_numerical_audit.py',
             'src/matlab/e11a_capture.m','src/matlab/e11a_run.m','experiments/11a_native_drilling.py',
             'tests/test_drilling_reproduction.py']
    immutable(out/'source_hashes.json',{name:sha(root/name) for name in sources})
    return config,out,archive

def run(root):
    root=Path(root).resolve();config,out,archive=freeze(root)
    for case in cases():
        for phase in config['repetitions']:
            folder=out/case['case_id']/phase;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'execution.json').exists():
                done=read(folder/'execution.json')
                assert done['request_sha256']==sha(folder/'request.json')
                for name,h in done['artifacts_sha256'].items():assert sha(folder/name)==h,name
                print(case['case_id'],phase,'preserved existing execution',flush=True);continue
            if (folder/'request.json').exists() or (folder/'native.mat').exists():
                raise RuntimeError(f'Interrupted evidence retained at {folder}; inspect, never overwrite')
            execution=Path(tempfile.mkdtemp(prefix=f"e11a_{case['case_id']}_{phase}_",dir='/private/tmp'))
            namespace='' if case['case_id']=='P0' else 'Chart 4 ICs/'
            names=DEPENDENCIES+([MAIN] if case['case_id']=='P0' else ['IC_Test.m','Chart_Plot.m'])
            copies={}
            with zipfile.ZipFile(archive) as z:
                for name in names:
                    target=execution/name;target.write_bytes(z.read(PREFIX+namespace+name));target.chmod(0o444)
                    copies[name]=sha(target)
            request=dict(case,execution_directory=str(execution),output=str(folder),seed=config['seed'],
                         copies_sha256=copies,source_namespace=namespace)
            write(folder/'request.json',request)
            quote=lambda p:str(p).replace("'","''")
            command=f"addpath('{quote(root/'src/matlab')}'); e11a_run('{quote(folder/'request.json')}');"
            env=dict(os.environ,MATLAB_PREFDIR=str(execution/'preferences'))
            started=time.perf_counter();timeout=False
            with (folder/'process.log').open('w') as log:
                try:
                    p=subprocess.run([config['matlab'],'-singleCompThread','-sd',str(execution),'-batch',command],
                        stdout=log,stderr=subprocess.STDOUT,env=env,timeout=config['maximum_seconds_per_process'])
                    code=p.returncode
                except subprocess.TimeoutExpired:timeout=True;code=None
            for name,h in copies.items():assert sha(execution/name)==h,name
            status=read(folder/'native_status.json') if (folder/'native_status.json').exists() else None
            metadata={'process_returncode':code,'timeout':timeout,'process_seconds':time.perf_counter()-started,
                'native_status':status,'request_sha256':sha(folder/'request.json'),'unchanged_source_copies':True}
            metadata['artifacts_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
            write(folder/'execution.json',metadata)
            print(case['case_id'],phase,'returncode',code,'native',None if status is None else
                  status['native_entrypoint_completed'],'seconds',round(metadata['process_seconds'],3),flush=True)
    print('Twelve prescribed execution attempts completed; numerical audit is separate.',flush=True)
