"""Static mechanical identities for E11C; no time integrator or controller.

The native, virtual-work, reaction-Schur and block-equation expressions are
diagnostics on identical operands. None is installed as a replacement plant.
"""
import gmpy2 as g
import numpy as np
from src.drilling_static_interval import Interval,transpose,multiply,solve as interval_solve


def exact(value):
    n,d=float(value).as_integer_ratio()
    return g.mpq(n,d)


def matrix(values,precision=None):
    return [[exact(x) if precision is None else Interval(float(x),precision)
             for x in row] for row in np.atleast_2d(values)]


def zero(a,n,m):
    z=a[0][0]*0
    return [[z for _ in range(m)] for _ in range(n)]


def add(a,b):
    if (len(a),len(a[0]))!=(len(b),len(b[0])):raise ValueError('Shape mismatch')
    return [[x+y for x,y in zip(r,s)] for r,s in zip(a,b)]


def neg(a):return [[-x for x in row] for row in a]
def subtract(a,b):return add(a,neg(b))
def horizontal(*blocks):return [sum((list(b[i]) for b in blocks),[]) for i in range(len(blocks[0]))]
def columns(a,start,stop):return [row[start:stop] for row in a]


def exact_solve(a,b):
    """Exact rational elimination with one pivot sequence for all RHS columns."""
    n=len(a);width=len(b[0]);x=[list(row)+list(rhs) for row,rhs in zip(a,b)]
    if any(len(row)!=n for row in a) or len(b)!=n:raise ValueError('Shape mismatch')
    for k in range(n):
        pivot=max(range(k,n),key=lambda i:abs(x[i][k]))
        x[k],x[pivot]=x[pivot],x[k]
        if x[k][k]==0:raise ArithmeticError('Singular exact system')
        for i in range(k+1,n):
            if x[i][k]==0:continue
            factor=x[i][k]/x[k][k]
            for j in range(k+1,n+width):x[i][j]-=factor*x[k][j]
            x[i][k]=g.mpq(0)
    result=[[g.mpq(0) for _ in range(width)] for _ in range(n)]
    for i in reversed(range(n)):
        for j in range(width):
            result[i][j]=(x[i][n+j]-sum(x[i][k]*result[k][j] for k in range(i+1,n)))/x[i][i]
    return result


def exact_rank(a):
    a=[list(row) for row in a];row=0
    for col in range(len(a[0])):
        pivot=next((i for i in range(row,len(a)) if a[i][col]!=0),None)
        if pivot is None:continue
        a[row],a[pivot]=a[pivot],a[row]
        for i in range(row+1,len(a)):
            if a[i][col]==0:continue
            f=a[i][col]/a[row][col]
            for j in range(col,len(a[0])):a[i][j]-=f*a[row][j]
        row+=1
        if row==len(a):break
    return row


def solve(a,b):
    return interval_solve(a,b) if isinstance(a[0][0],Interval) else exact_solve(a,b)


def record(a):
    return [[x.record() if isinstance(x,Interval) else
             [str(x.numerator),str(x.denominator)] for x in row] for row in a]


def approximate(a):
    return np.array([[float(x.midpoint()) if isinstance(x,Interval) else float(x)
                      for x in row] for row in a])


def maximum(a):
    if isinstance(a[0][0],Interval):
        return max(max(abs(float(x.lo)),abs(float(x.hi))) for row in a for x in row)
    return max(float(abs(x)) for row in a for x in row)


def identity_certificate(a,scale=1.):
    if isinstance(a[0][0],Interval):
        enclosed=all(x.lo<=0<=x.hi for row in a for x in row)
        return {'contains_zero':enclosed,'maximum_absolute_enclosure_display':maximum(a),
                'maximum_scaled_enclosure_display':maximum(a)/max(1.,scale),
                'zero_justification':'Algebraic identity conditional on validated invertibility; interval overlap alone is not a zero proof.'}
    return {'exact_zero':all(x==0 for row in a for x in row),'maximum_absolute_display':maximum(a)}


def ldl_positive(a):
    """Positivity certificate for the symmetric structured real assembly only.

    Archived floating matrices are not silently symmetrized. Symmetry here
    follows from the real congruence assembly, despite endpoint rounding.
    """
    n=len(a);lower=zero(a,n,n);pivots=[]
    for i in range(n):
        d=a[i][i]-sum(lower[i][k]*lower[i][k]*pivots[k] for k in range(i))
        if d.lo<=0:
            return {'status':'UNRESOLVED','failed_pivot':i,'pivots':[x.record() for x in pivots],'candidate_pivot':d.record()}
        pivots.append(d)
        for j in range(i+1,n):
            lower[j][i]=(a[j][i]-sum(lower[j][k]*lower[i][k]*pivots[k] for k in range(i)))/d
    return {'status':'SUPPORTED','pivots':[x.record() for x in pivots],
            'scope':'Symmetric structured-real matrix; not an SPD assertion about nonsymmetric archived dyadics.'}


def constrained_systems(m,k,d,t,s,c,w):
    """Evaluate three ideal-reaction formulations on identical static RHS."""
    n,nq,nc=len(m),len(t[0]),len(c);ct=transpose(c);tt=transpose(t)
    solved=solve(m,horizontal(w,ct));u=columns(solved,0,len(w[0]));mc=columns(solved,len(w[0]),len(solved[0]))
    compatibility=multiply(c,u)
    reaction_metric=multiply(c,mc)
    lam=solve(reaction_metric,neg(compatibility));reaction=multiply(ct,lam)
    corrected=add(u,multiply(mc,lam))
    mh=multiply(tt,multiply(m,t));kh=multiply(tt,multiply(k,t));dh=multiply(tt,multiply(d,t))
    vw=solve(mh,multiply(tt,w))
    block=horizontal(m,neg(ct))+horizontal(c,zero(m,nc,nc))
    block_rhs=w+zero(m,nc,len(w[0]));block_solution=solve(block,block_rhs)
    block_a=block_solution[:n];block_lam=block_solution[n:]
    retained=multiply(s,u);delta=multiply(s,multiply(mc,lam))
    checks={
        'ST_identity':subtract(multiply(s,t),[[t[0][0]*0+(1 if i==j else 0) for j in range(nq)] for i in range(nq)]),
        'CT_identity':multiply(c,t),
        'constraint_acceleration':multiply(c,corrected),
        'reaction_virtual_work':multiply(tt,reaction),
        'full_force_balance':subtract(subtract(multiply(m,corrected),w),reaction),
        'virtual_work_vs_reaction':subtract(multiply(s,corrected),vw),
        'virtual_work_lift':subtract(corrected,multiply(t,vw)),
        'block_vs_reaction_acceleration':subtract(block_a,corrected),
        'block_vs_reaction_multiplier':subtract(block_lam,lam),
        'retained_reaction_effect':subtract(add(retained,delta),vw)}
    scale=max(maximum(w),maximum(reaction),maximum(corrected),maximum(vw),1.)
    certificates={name:identity_certificate(value,scale) for name,value in checks.items()}
    return {'M':m,'K':k,'D':d,'T':t,'S':s,'C':c,'W':w,'M_H':mh,'K_H':kh,'D_H':dh,
            'unreacted_full_acceleration':u,'native_retained_acceleration':retained,
            'native_constraint_acceleration':compatibility,'reaction_metric':reaction_metric,
            'lambda':lam,'reaction':reaction,'ideal_full_acceleration':corrected,
            'retained_reaction_acceleration':delta,'virtual_work_acceleration':vw,
            'block_acceleration':block_a,'block_lambda':block_lam,
            'block_matrix':block,'identity_certificates':certificates}


def force_blocks(m,k,d,t,gu,gb,gravity,snapshots,ic,vs,nominal_force):
    """25 prescribed operator columns, four snapshots, two reference times."""
    kt,dt=multiply(k,t),multiply(d,t)
    ops=horizontal(kt,dt,gu,gb,gravity)
    ws=[]
    for q,v,f in snapshots:
        ws.append(subtract(subtract(f,multiply(kt,q)),multiply(dt,v)))
    for time in (0,20):
        q=add(ic,[[row[0]*time] for row in vs])
        ws.append(subtract(subtract(nominal_force,multiply(kt,q)),multiply(dt,vs)))
    return horizontal(ops,*ws)
