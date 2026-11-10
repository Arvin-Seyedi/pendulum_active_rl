"""Fixed supervised B0 learner/baseline. No RL objectives or simulator calls."""
from pathlib import Path
import hashlib,time,resource
import numpy as np
import torch
from torch import nn
from src.ensemble import configure_determinism
from src.quadrotor_mmaq_v1 import safe_write

class ArchivePolicy(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(496,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2),nn.Sigmoid())
    def forward(self,x):return self.network(x)

def parameters_sha(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def uniform_schedule(seed,updates,batch):
    return np.random.Generator(np.random.PCG64(seed+1000000)).random((updates,batch))

def predict(model,x):
    model.eval()
    with torch.no_grad():return torch.cat([model(torch.as_tensor(a,dtype=torch.float32)) for a in np.array_split(np.asarray(x),max(1,(len(x)+1023)//1024))]).cpu().numpy().astype(np.float64)

def fit_fixed(x,y,seed,config,folder,resource_check=lambda:None):
    folder=Path(folder)
    if folder.exists():raise FileExistsError('Retain completed/failed fit; no overwrite')
    folder.mkdir(parents=True);configure_determinism(seed)
    model=ArchivePolicy().cpu();initial=parameters_sha(model);optimizer=torch.optim.Adam(model.parameters(),lr=config['optimizer']['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
    xx=torch.as_tensor(np.asarray(x),dtype=torch.float32);yy=torch.as_tensor(np.asarray(y),dtype=torch.float32)
    assert xx.shape==(len(yy),496) and yy.shape[1:]==(2,) and torch.isfinite(xx).all() and torch.isfinite(yy).all()
    uniforms=uniform_schedule(seed,config['max_updates'],config['batch_size']);indices=np.floor(uniforms*len(xx)).astype(np.int64);history=[]
    torch.save({'model':model.state_dict(),'seed':seed},folder/'initial.pt')
    start=time.process_time();wall=time.perf_counter()
    try:
        for k,ids in enumerate(indices,1):
            model.train();optimizer.zero_grad(set_to_none=True);loss=((model(xx[ids])-yy[ids])**2).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            grad=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))) if k==1 or k%config['history_interval']==0 else None
            optimizer.step()
            if k==1 or k%config['history_interval']==0:
                resource_check();row={'update':k,'batch_MSE':float(loss.detach()),'gradient_norm':grad};history.append(row)
                safe_write(folder/'history.json',history);safe_write(folder/'progress.json',{'updates':k,'cpu_seconds':time.process_time()-start,'wall_seconds':time.perf_counter()-wall,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite final parameters')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'seed':seed,'updates':config['max_updates']},folder/'final.pt')
        result={'complete':True,'seed':seed,'updates':config['max_updates'],'labels':len(xx),'initial_parameters_sha256':initial,'final_parameters_sha256':parameters_sha(model),'uniform_schedule_sha256':hashlib.sha256(uniforms.tobytes()).hexdigest(),'index_schedule_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),'history':history,'CPU_seconds':time.process_time()-start,'wall_seconds':time.perf_counter()-wall}
        safe_write(folder/'completion.json',result);return model,result
    except BaseException as exc:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'last_update':k},folder/'failed_partial.pt')
        safe_write(folder/'failure.json',{'kind':type(exc).__name__,'message':str(exc),'last_update':k,'complete':False});raise

def ridge_fit(x,y,regularization):
    design=np.c_[np.ones(len(x)),np.asarray(x,dtype=np.float64)];y=np.asarray(y,dtype=np.float64)
    # Explicit contractions avoid the pinned matmul path's reproduced floating-
    # status warnings. Objective, operands and regularization are unchanged.
    A=np.einsum('ni,nj->ij',design,design,optimize=False)/len(design);A[1:,1:]+=regularization*np.eye(x.shape[1]);b=np.einsum('ni,nj->ij',design,y,optimize=False)/len(design)
    beta=np.linalg.solve(A,b);residual=float(np.linalg.norm(np.einsum('ij,jk->ik',A,beta,optimize=False)-b)/(np.linalg.norm(A)*np.linalg.norm(beta)+np.linalg.norm(b)))
    if not np.isfinite(beta).all() or residual>1e-12:raise FloatingPointError('Ridge backward error failed')
    return beta,{'backward_error':residual,'condition_number':float(np.linalg.cond(A))}

def ridge_predict(beta,x):
    raw=np.einsum('ni,ij->nj',np.c_[np.ones(len(x)),np.asarray(x,dtype=np.float64)],beta,optimize=False)
    return np.clip(raw,0,1),raw
