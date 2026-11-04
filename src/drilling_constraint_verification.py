"""Independent E11C checker using Python fractions, not MPFR/producer solves.

Exact block residuals and an integer Bareiss determinant certify uniqueness of
the archived dyadic block solution. A separate fraction/dyadic outward-rounded
LDL check verifies positivity of the declared structured-real interval matrices.
"""
from fractions import Fraction as F
from pathlib import Path
import math
import time
import json
import numpy as np
from src.drilling_reproduction import read,write,sha
from src.drilling_archive_verification import check_hashes


def q(pair):return F(int(pair[0]),int(pair[1]))
def decode(a):return [[q(x) for x in row] for row in a]
def trans(a):return [list(row) for row in zip(*a)]
def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def sub(a,b):return [[x-y for x,y in zip(r,s)] for r,s in zip(a,b)]
def plus(a,b):return [[x+y for x,y in zip(r,s)] for r,s in zip(a,b)]
def allzero(a):return all(x==0 for row in a for x in row)
def cols(a,lo,hi):return [row[lo:hi] for row in a]
def cat(*a):return [sum((b[i] for b in a),[]) for i in range(len(a[0]))]


def bareiss_determinant(a):
    """Fraction-free determinant after exact integer row scaling."""
    integer=[];scales=[]
    for row in a:
        den=math.lcm(*(x.denominator for x in row));scales.append(den)
        integer.append([int(x*den) for x in row])
    n=len(integer);previous=1;sign=1
    for k in range(n-1):
        pivot=next((i for i in range(k,n) if integer[i][k]),None)
        if pivot is None:return F(0)
        if pivot!=k:integer[k],integer[pivot]=integer[pivot],integer[k];sign=-sign
        value=integer[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                numerator=integer[i][j]*value-integer[i][k]*integer[k][j]
                if numerator%previous:raise AssertionError('Non-exact Bareiss division')
                integer[i][j]=numerator//previous
            integer[i][k]=0
        previous=value
    return F(sign*integer[-1][-1],math.prod(scales))


def directed(x,p,upper):
    """Round a fraction outward to p significant binary bits, no MPFR."""
    if x==0:return F(0)
    n,d=abs(x.numerator),x.denominator;e=n.bit_length()-d.bit_length()
    power=F(2**e) if e>=0 else F(1,2**(-e))
    if abs(x)<power:e-=1
    shift=e-p+1;step=F(2**shift) if shift>=0 else F(1,2**(-shift))
    scaled=x/step
    count=-((-scaled.numerator)//scaled.denominator) if upper else scaled.numerator//scaled.denominator
    return count*step


def ibin(a,b,op,p):
    if op=='+':lo,hi=a[0]+b[0],a[1]+b[1]
    elif op=='-':lo,hi=a[0]-b[1],a[1]-b[0]
    else:
        if op=='/' and b[0]<=0<=b[1]:raise ArithmeticError('Independent interval division unresolved')
        values=[x*y if op=='*' else x/y for x in a for y in b];lo,hi=min(values),max(values)
    return directed(lo,p,False),directed(hi,p,True)


def fraction_ldl(a,p):
    a=[[(q(x['lo']),q(x['hi'])) for x in row] for row in a];n=len(a)
    low=[[(F(0),F(0)) for _ in range(n)] for _ in range(n)];piv=[]
    for i in range(n):
        di=a[i][i]
        for k in range(i):di=ibin(di,ibin(ibin(low[i][k],low[i][k],'*',p),piv[k],'*',p),'-',p)
        if di[0]<=0:return {'positive':False,'unresolved_pivot':i}
        piv.append(di)
        for j in range(i+1,n):
            num=a[j][i]
            for k in range(i):num=ibin(num,ibin(ibin(low[j][k],low[i][k],'*',p),piv[k],'*',p),'-',p)
            low[j][i]=ibin(num,di,'/',p)
    return {'positive':True,'smallest_pivot_lower_display':float(min(x[0] for x in piv))}


def interval_matrix(a):return [[(q(x['lo']),q(x['hi'])) for x in row] for row in a]


def imm(a,b,p):
    result=[]
    for row in a:
        rr=[]
        for col in zip(*b):
            value=(F(0),F(0))
            for x,y in zip(row,col):value=ibin(value,ibin(x,y,'*',p),'+',p)
            rr.append(value)
        result.append(rr)
    return result


def isub(a,b,p):return [[ibin(x,y,'-',p) for x,y in zip(r,s)] for r,s in zip(a,b)]
def zero_enclosed(a):return all(x[0]<=0<=x[1] for row in a for x in row)


def verify(root,out):
    started=time.perf_counter();root,out=Path(root),Path(out)
    check_hashes(out,read(out/'completion.json')['artifact_sha256'])
    cfg=read(out/'protocol.json');raw=read(out/'exact_dyadic.json')
    keys=['M','K','D','T','S','C','Gu','Gb','gravity','IC','Vs','W','M_H','K_H','D_H',
          'unreacted_full_acceleration','native_retained_acceleration','lambda','reaction',
          'ideal_full_acceleration','retained_reaction_acceleration','virtual_work_acceleration',
          'block_acceleration','block_lambda','block_matrix']
    x={k:decode(raw[k]) for k in keys};m,t,c,s=x['M'],x['T'],x['C'],x['S']
    frozen=dict(np.load(out/'frozen_binary64.npz'))
    cv=lambda a:[[F.from_float(float(v)) for v in row] for row in np.atleast_2d(a)]
    for key,name in [('M','M1_global'),('K','K1_global'),('D','D1_global'),('IC','IC'),('Vs','Vs'),('gravity','Gravity')]:
        assert x[key]==cv(frozen[name]),name
    assert t==cv(np.load(out/'frozen_T.npy'))
    gu=np.zeros((20,2));gb=np.zeros((20,2))
    gu[:8,0]=frozen['TopR'][:,0];gu[:8,1]=frozen['TopR'][:,2]
    gb[-8:,0]=-frozen['BotR'][:,4];gb[-8:,1]=-frozen['BotR'][:,6]
    assert x['Gu']==cv(gu) and x['Gb']==cv(gb)
    identity=[[F(i==j) for j in range(10)] for i in range(10)]
    assert mm(s,t)==identity and allzero(mm(c,t))
    assert x['M_H']==mm(trans(t),mm(m,t))
    assert x['K_H']==mm(trans(t),mm(x['K'],t))
    assert x['D_H']==mm(trans(t),mm(x['D'],t))
    assert allzero(sub(mm(m,x['unreacted_full_acceleration']),x['W']))
    assert allzero(mm(c,x['ideal_full_acceleration']))
    assert allzero(mm(trans(t),x['reaction']))
    assert allzero(sub(sub(mm(m,x['ideal_full_acceleration']),x['W']),x['reaction']))
    assert allzero(sub(mm(s,x['ideal_full_acceleration']),x['virtual_work_acceleration']))
    assert allzero(sub(mm(t,x['virtual_work_acceleration']),x['ideal_full_acceleration']))
    assert x['reaction']==mm(trans(c),x['lambda'])
    assert allzero(sub(x['ideal_full_acceleration'],x['block_acceleration']))
    assert x['block_lambda']==x['lambda']
    nzero=[[F(0) for _ in range(10)] for _ in range(10)]
    expected_block=cat(m,[[-v for v in row] for row in trans(c)])+cat(c,nzero)
    assert x['block_matrix']==expected_block
    determinant=bareiss_determinant(expected_block);assert determinant!=0
    # Reconstruct the 25 fixed operator columns and six frozen-state/reference RHS.
    kt,dt=mm(x['K'],t),mm(x['D'],t);op=cat(kt,dt,x['Gu'],x['Gb'],x['gravity'])
    assert cols(x['W'],0,25)==op
    manifest=read(out/'snapshot_manifest.json');samples=[]
    for spec in manifest:
        assert sha(out/spec['file'])==spec['sha256']
        with np.load(out/spec['file']) as sample:
            samples.append({k:sample[k].copy() for k in sample.files})
    for i,sample in enumerate(samples):
        w=sub(sub(cv(sample['F']),mm(kt,cv(sample['U'][:10]))),mm(dt,cv(sample['U'][10:])))
        assert cols(x['W'],25+i,26+i)==w
    for i,time_value in enumerate((0,20)):
        ref=plus(x['IC'],[[time_value*row[0]] for row in x['Vs']])
        w=sub(sub(cv(samples[0]['F']),mm(kt,ref)),mm(dt,x['Vs']))
        assert cols(x['W'],29+i,30+i)==w
    positivity={};overlaps=0;previous=None;interval_checks=0
    for p in cfg['precisions_bits']:
        r=read(out/f'interval_{p}.json')
        positivity[str(p)]={k:fraction_ldl(r[k],p) for k in ('M','M_H','K_H')}
        for name in ('M','M_H','K_H'):
            assert positivity[str(p)][name]['positive']==(r['positivity'][name]['status']=='SUPPORTED')
        im={key:interval_matrix(r[key]) for key in ('M','C','T','W','reaction','ideal_full_acceleration','M_H','virtual_work_acceleration')}
        checks=[imm(im['C'],im['ideal_full_acceleration'],p),imm(trans(im['T']),im['reaction'],p),
                isub(isub(imm(im['M'],im['ideal_full_acceleration'],p),im['W'],p),im['reaction'],p),
                isub(imm(im['M_H'],im['virtual_work_acceleration'],p),imm(trans(im['T']),im['W'],p),p)]
        for value in checks:
            assert zero_enclosed(value);interval_checks+=sum(len(row) for row in value)
        if previous:
            for key in ('native_constraint_acceleration','virtual_work_acceleration','lambda','native_K_Vs','ideal_K_Vs'):
                for row_a,row_b in zip(previous[key],r[key]):
                    for va,vb in zip(row_a,row_b):
                        al,ah=q(va['lo']),q(va['hi']);bl,bh=q(vb['lo']),q(vb['hi'])
                        assert max(al,bl)<=min(ah,bh);overlaps+=1
        previous=r
    for name,h in read(out/'source_identities.json')['implementation_sha256'].items():assert sha(root/name)==h,name
    for name,h in read(out/'preservation_before.json').items():assert sha(root/name)==h,name
    result={'passed':True,'exact_rhs_columns_checked':31,'exact_block_determinant_nonzero':True,
        'block_determinant_exact':[str(determinant.numerator),str(determinant.denominator)],
        'fraction_interval_scalar_residuals_checked':interval_checks,'cross_precision_scalar_overlaps':overlaps,
        'independent_fraction_LDL':positivity,'new_trajectories':0,'producer_math_imported':False,
        'seconds':time.perf_counter()-started}
    write(out/'independent_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='block_determinant_exact'},indent=2))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    verify(root,args.output or root/read(root/'configs/experiment_11c.json')['output'])
