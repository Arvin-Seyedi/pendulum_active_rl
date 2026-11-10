"""Prospectively fixed B1 networks and continuation checkpoints."""
from pathlib import Path
import hashlib,time,resource
import numpy as np
import torch
from torch import nn
from src.ensemble import configure_determinism
from src.quadrotor_archive_learner import uniform_schedule,parameters_sha,predict,ridge_fit,ridge_predict
from src.quadrotor_mmaq_v1 import safe_write

class AdequacyPolicy(nn.Module):
    def __init__(self,dimension,seed):
        super().__init__();self.network=nn.Sequential(nn.Linear(dimension,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2),nn.Sigmoid())
        self.layer_seeds=[int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(3)]
        for layer,seed in zip([self.network[0],self.network[2],self.network[4]],self.layer_seeds):
            with torch.random.fork_rng():torch.manual_seed(seed);layer.reset_parameters()
    def forward(self,x):return self.network(x)

def fit(x,y,seed,c,folder,check=lambda:None):
    folder=Path(folder)
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True);configure_determinism(seed);net=AdequacyPolicy(x.shape[1],seed)
    op=torch.optim.Adam(net.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
    xx=torch.as_tensor(x,dtype=torch.float32);yy=torch.as_tensor(y,dtype=torch.float32)
    if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):raise ValueError('Nonfinite training data')
    uniform=uniform_schedule(seed,c['max_updates'],c['batch_size']);indices=np.floor(uniform*len(x)).astype(np.int64)
    initial=parameters_sha(net);torch.save({'model':net.state_dict()},folder/'initial.pt');history=[];start=time.process_time();wall=time.perf_counter();k=0
    try:
        for k,ids in enumerate(indices,1):
            op.zero_grad(set_to_none=True);loss=(net(xx[ids])-yy[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if not all(torch.isfinite(v.grad).all() for v in net.parameters()):raise FloatingPointError('Nonfinite gradient')
            logged=k==1 or k%c['history_interval']==0
            grad=float(torch.sqrt(sum(v.grad.square().sum() for v in net.parameters()))) if logged else None
            op.step()
            if logged:
                check();history.append({'update':k,'batch_MSE':float(loss.detach()),'gradient_norm':grad});safe_write(folder/'history.json',history)
            if k in c['checkpoints']:
                assert all(torch.isfinite(v).all() for v in net.parameters())
                torch.save({'model':net.state_dict(),'optimizer':op.state_dict(),'updates':k,'seed':seed},folder/f'update_{k}.pt')
        result={'complete':True,'updates':k,'seed':seed,'dimension':x.shape[1],'labels':len(x),'parameters':sum(v.numel() for v in net.parameters()),'layer_seeds':net.layer_seeds,
            'initial_parameters_sha256':initial,'final_parameters_sha256':parameters_sha(net),'uniform_schedule_sha256':hashlib.sha256(uniform.tobytes()).hexdigest(),
            'index_schedule_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),'CPU_seconds':time.process_time()-start,'wall_seconds':time.perf_counter()-wall,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        safe_write(folder/'completion.json',result);return result
    except BaseException as exc:
        torch.save({'model':net.state_dict(),'optimizer':op.state_dict(),'last_attempted_update':k},folder/'failed_partial.pt')
        safe_write(folder/'failure.json',{'complete':False,'kind':type(exc).__name__,'message':str(exc),'last_attempted_update':k});raise
