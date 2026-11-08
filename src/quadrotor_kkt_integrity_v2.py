"""Separately versioned canonical KKT checks; never solves an optimization.

For l <= g(z) <= u, L=f+lambda.T*g and lambda=mu_upper-mu_lower.
Absent constraint sides have NO multiplier or complementarity product. Wrong
one-sided signs are a separate residual, not an infinite-side multiplication.
Historical v1 calculations and classifications are not overwritten.
"""
import numpy as np

VERSION = 'canonical_bound_kkt_v2'


def _max(values):
    return float(np.max(values, initial=0.))


def canonical_kkt(values, lower, upper, multipliers, lagrangian_gradient):
    """Return raw per-row diagnostics and separate scalar residuals.

    Equality multipliers are free. For inequalities the nonnegative multipliers
    are the supported parts of signed lambda; unsupported signs are reported.
    Fully unbounded rows require lambda=0. NaN, malformed bounds and nonfinite
    iterates/derivatives fail closed. Inf is legitimate only as an absent side.
    """
    g,lo,hi,lam = [np.asarray(v,dtype=float).reshape(-1) for v in
                    (values,lower,upper,multipliers)]
    grad=np.asarray(lagrangian_gradient,dtype=float).reshape(-1)
    raw={'values':g,'lower':lo,'upper':hi,'multipliers':lam,'lagrangian_gradient':grad}
    reasons=[]
    if not (g.shape==lo.shape==hi.shape==lam.shape):reasons.append('constraint_shape_mismatch')
    if not np.isfinite(g).all():reasons.append('nonfinite_constraint_values')
    if not np.isfinite(lam).all():reasons.append('nonfinite_multipliers')
    if not np.isfinite(grad).all():reasons.append('nonfinite_lagrangian_gradient')
    if np.isnan(lo).any() or np.isnan(hi).any():reasons.append('NaN_bound')
    if np.isposinf(lo).any() or np.isneginf(hi).any():reasons.append('invalid_infinite_bound_side')
    if not reasons and np.any(lo>hi):reasons.append('inverted_bounds')
    if reasons:
        return {'version':VERSION,'valid_arithmetic':False,'invalid_reasons':reasons,'rows':raw,
                'metrics_are_rejection_sentinels':True,
                'metrics':{'primal_inf':np.inf,'stationarity_inf':np.inf,
                           'dual_sign_inf':np.inf,'complementarity_inf':np.inf,
                           'normalized_complementarity_inf':np.inf}}
    fl=np.isfinite(lo);fu=np.isfinite(hi);eq=fl&fu&(lo==hi)
    lower_only=fl&~fu;upper_only=fu&~fl;free=~fl&~fu
    ml=np.zeros_like(g);mu=np.zeros_like(g)
    ml[fl&~eq]=np.maximum(-lam[fl&~eq],0)
    mu[fu&~eq]=np.maximum(lam[fu&~eq],0)
    sign=np.zeros_like(g)
    sign[lower_only]=np.maximum(lam[lower_only],0)
    sign[upper_only]=np.maximum(-lam[upper_only],0)
    sign[free]=np.abs(lam[free])
    sl=np.full_like(g,np.nan);su=np.full_like(g,np.nan)
    pl=np.zeros_like(g);pu=np.zeros_like(g);cl=np.zeros_like(g);cu=np.zeros_like(g)
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            sl[fl]=g[fl]-lo[fl];su[fu]=hi[fu]-g[fu]
            pl[fl]=np.maximum(-sl[fl],0);pu[fu]=np.maximum(-su[fu],0)
            cl[fl&~eq]=ml[fl&~eq]*sl[fl&~eq]
            cu[fu&~eq]=mu[fu&~eq]*su[fu&~eq]
            normalized=np.maximum(np.abs(cl),np.abs(cu))/(1+np.abs(lam))
            metrics={'primal_inf':max(_max(pl),_max(pu)),
                     'stationarity_inf':_max(np.abs(grad)), 'dual_sign_inf':_max(sign),
                     'complementarity_inf':max(_max(np.abs(cl)),_max(np.abs(cu))),
                     'normalized_complementarity_inf':_max(normalized)}
    except FloatingPointError as exc:
        return {'version':VERSION,'valid_arithmetic':False,'invalid_reasons':[str(exc)],'rows':raw,
                'metrics_are_rejection_sentinels':True,
                'metrics':dict.fromkeys(['primal_inf','stationarity_inf','dual_sign_inf',
                                        'complementarity_inf','normalized_complementarity_inf'],np.inf)}
    raw.update(finite_lower=fl,finite_upper=fu,equality=eq,lower_slack=sl,upper_slack=su,
               lower_multiplier=ml,upper_multiplier=mu,dual_sign_residual=sign,
               lower_violation=pl,upper_violation=pu,lower_complementarity=cl,upper_complementarity=cu)
    return {'version':VERSION,'valid_arithmetic':True,'invalid_reasons':[],
            'metrics_are_rejection_sentinels':False,'rows':raw,'metrics':metrics}


def require_kkt(result,status,*,primal_tolerance=1e-4,required_status='Solve_Succeeded'):
    """Keep frozen numerical thresholds; add the previously missing sign check."""
    if status!=required_status:raise AssertionError('Solver status rejected: '+str(status))
    if not result['valid_arithmetic']:raise AssertionError('Invalid KKT arithmetic: '+str(result['invalid_reasons']))
    m=result['metrics']
    if not np.isfinite(list(m.values())).all():raise AssertionError('Nonfinite KKT residual')
    for name,limit in [('primal_inf',primal_tolerance),('stationarity_inf',1.01e-8),
                       ('dual_sign_inf',1.01e-8),('normalized_complementarity_inf',1.01e-8)]:
        if m[name]>limit:raise AssertionError(name+' exceeds unchanged tolerance')


def matching_program_values(own, native, tolerance=1e-10):
    """Compare finite values and exact signs of absent infinite bound sides."""
    if len(own)!=len(native):raise AssertionError('Program output counts differ')
    errors=[]
    for left,right in zip(own,native):
        a,b=np.asarray(left).reshape(-1),np.asarray(right).reshape(-1)
        if a.shape!=b.shape or np.isnan(a).any() or np.isnan(b).any():
            raise AssertionError('Invalid program reconstruction operands')
        if not (np.array_equal(np.isposinf(a),np.isposinf(b)) and
                np.array_equal(np.isneginf(a),np.isneginf(b))):
            raise AssertionError('Infinite bound signs differ')
        finite=np.isfinite(a)&np.isfinite(b)
        error=float(np.max(np.abs(a[finite]-b[finite]),initial=0));errors.append(error)
    if max(errors,default=0)>tolerance:raise AssertionError('Program reconstruction failed')
    return max(errors,default=0)


def nlp_readout(f,N):
    """Pure symbolic equations for archived vectors, no environment or solver.

    Reuses the frozen independent NLP construction with a minimal mathematical
    Opti shell, not a controller or plant. No optimizer is configured or called.
    """
    import casadi as c
    from types import SimpleNamespace
    from src.quadrotor_mmaq_verification_v1 import independent_nlp_program
    # The historical constructor requires only these Opti symbol/dimension fields
    # at its final native-program comparison. Supply a symbolic placeholder,
    # then use only its independently assembled first returned function.
    nv=8*N+6;np_=6*N+12;ng=22*N+18
    z=c.MX.sym('shell_z',nv);p=c.MX.sym('shell_p',np_)
    shell=SimpleNamespace(nx=nv,np=np_,ng=ng,x=z,p=p,f=c.MX(0),g=c.MX.zeros(ng),
                          lbg=c.MX.zeros(ng),ubg=c.MX.zeros(ng))
    ctrl=SimpleNamespace(T=N,opti_dict={'opti':shell})
    independent,_=independent_nlp_program(ctrl,f)
    return independent
