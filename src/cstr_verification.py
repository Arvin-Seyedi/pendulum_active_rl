"""Independent nominal CSTR algebra, Radau NLP and raw numerical diagnostics."""
import numpy as np
from src.quadrotor_kkt_integrity_v2 import canonical_kkt

LOW=np.array([.1,.1,50.,50.]);HIGH=np.array([2.,2.,140.,140.]);WIDTH=HIGH-LOW
ULO=np.array([5.,-8500.]);UHI=np.array([100.,0.]);UWIDTH=UHI-ULO
SX=np.array([1.,1.,100.,100.]);SU=np.array([100.,2000.]);DT=.005;N=20

def rhs(_t,x,u):
    """Time h, concentrations mol/L, temperatures C, F h^-1, Q kJ/h."""
    a,b,tr,tk=np.asarray(x);F,Q=np.asarray(u)
    k=1.287e12*np.exp(-9758.3/(tr+273.15));k3=9.043e9*np.exp(-8560./(tr+273.15))
    r1,r2,r3=k*a,k*b,k3*a*a
    return np.array([F*(5.1-a)-r1-r3,r1-r2-F*b,
        -(4.2*r1-11.*r2-41.85*r3)/(.9342*3.01)+F*(130.-tr)+866.88*(tk-tr)/(.9342*3.01*10.01),
        (Q+866.88*(tr-tk))/10.])

def symbolic_rhs(x,u):
    import casadi as c
    a,b,tr,tk=c.vertsplit(x);F,Q=c.vertsplit(u)
    k=1.287e12*c.exp(-9758.3/(tr+273.15));k3=9.043e9*c.exp(-8560./(tr+273.15))
    r1,r2,r3=k*a,k*b,k3*a*a
    return c.vertcat(F*(5.1-a)-r1-r3,r1-r2-F*b,
        -(4.2*r1-11.*r2-41.85*r3)/(.9342*3.01)+F*(130.-tr)+866.88*(tk-tr)/(.9342*3.01*10.01),
        (Q+866.88*(tr-tk))/10.)

def independent_program():
    """Reconstruct the source's 376-variable layout, not its discretizer calls."""
    import casadi as c
    z=c.SX.sym('independent_z',376);p=c.SX.sym('independent_p',6);lam=c.SX.sym('lambda_g',324)
    X=lambda k,j:z[16*k+4*j:16*k+4*j+4]
    U=lambda k:z[336+2*k:338+2*k]
    ell=lambda x:(x[0]-.7)**2+(x[1]-.6)**2
    cons=[X(0,3)-p[:4]/SX];cost=c.SX(0)
    for k in range(N):
        x0,x1,x2,aux,end=X(k,3),X(k+1,0),X(k+1,1),X(k+1,2),X(k+1,3)
        cons.extend([DT*symbolic_rhs(x1*SX,U(k)*SU)/SX-(-2*x0+1.5*x1+.5*x2),
                     DT*symbolic_rhs(x2*SX,U(k)*SU)/SX-(2*x0-4.5*x1+2.5*x2),
                     aux-x2,aux-end])
        previous=p[4:]/SU if k==0 else U(k-1);du=U(k)-previous
        cost+=ell(x0*SX)+.1*du[0]**2+.001*du[1]**2
    cost+=ell(X(N,3)*SX);g=c.vertcat(*cons);grad=c.gradient(cost,z);jlam=c.jtimes(g,z,lam,True)
    return c.Function('independent_CSTR_NLP',[z,p,lam],[cost,g,grad,jlam])

def graph_bounds():
    lower=np.full(376,-np.inf);upper=np.full(376,np.inf)
    for k in range(1,20):
        for j in range(4):lower[k*16+j*4:k*16+j*4+4]=LOW/SX;upper[k*16+j*4:k*16+j*4+4]=HIGH/SX
    lower[336:]=np.tile(ULO/SU,20);upper[336:]=np.tile(UHI/SU,20)
    return lower,upper,np.zeros(324),np.zeros(324)

def candidate_metrics(candidate,program):
    z,p,lg,lx=[np.asarray(candidate[k],dtype=float).reshape(-1) for k in ['z','p','lam_g','lam_x']]
    if z.shape!=(376,) or p.shape!=(6,) or lg.shape!=(324,) or lx.shape!=(376,):raise ValueError('Invalid candidate shape')
    if not all(np.isfinite(v).all() for v in [z,p,lg,lx]):raise FloatingPointError('Nonfinite candidate')
    f,g,df,jl=[np.array(v).reshape(-1) for v in program(z,p,lg)]
    lower,upper,gl,gu=graph_bounds()
    for name,expected in [('lbx',lower),('ubx',upper),('lbg',gl),('ubg',gu)]:
        a=np.asarray(candidate[name]).reshape(-1)
        if a.tobytes()!=expected.tobytes():raise AssertionError('Native graph bound mismatch: '+name)
    # Coordinates normalized by physical constraint widths; objective unchanged.
    qz=np.r_[np.tile(WIDTH/SX,84),np.tile(UWIDTH/SU,20)];qg=np.tile(WIDTH/SX,81)
    stationarity=qz*(df+jl+lx);denom=1+max(np.max(abs(qz*df)),np.max(abs(qz*jl)),np.max(abs(qz*lx)))
    kkt=canonical_kkt(np.r_[z/qz,g/qg],np.r_[lower/qz,gl/qg],np.r_[upper/qz,gu/qg],np.r_[lx*qz,lg*qg],stationarity/denom)
    raw=kkt['metrics'];f_given=float(np.asarray(candidate['f']).reshape(-1)[0]);g_given=np.asarray(candidate['g']).reshape(-1)
    return {'valid_arithmetic':kkt['valid_arithmetic'],'primal_scaled_inf':raw['primal_inf'],
        'scaled_stationarity_inf':raw['stationarity_inf'],'scaled_complementarity_inf':raw['normalized_complementarity_inf'],
        'dual_sign_inf':raw['dual_sign_inf'],'objective':float(f[0]),'objective_reconstruction_error':abs(f_given-f[0]),
        'constraint_reconstruction_error':float(np.max(abs(g_given-g))),
        'initial_state_parameter_error':float(np.max(abs(z[12:16]*SX-p[:4]))),
        'command':(z[336:338]*SU).tolist(),'stationarity_denominator':float(denom)}

def require_candidate(metrics,stats,config):
    if not stats.get('success') or stats.get('return_status')!=config['status_required']:raise AssertionError('Native solver status rejected')
    if not metrics['valid_arithmetic']:raise AssertionError('Invalid KKT arithmetic')
    for name,tol in [('primal_scaled_inf',config['primal_tolerance']),('scaled_stationarity_inf',config['stationarity_tolerance']),
        ('scaled_complementarity_inf',config['complementarity_tolerance']),('dual_sign_inf',config['dual_sign_tolerance']),
        ('objective_reconstruction_error',config['objective_tolerance']),('constraint_reconstruction_error',config['graph_tolerance'])]:
        if not np.isfinite(metrics[name]) or metrics[name]>tol:raise AssertionError(name+' acceptance failure')

def state_margins(states):
    x=np.asarray(states,dtype=float)
    if x.shape[-1]!=4 or not np.isfinite(x).all():raise ValueError('Nonfinite/invalid raw state')
    return np.minimum(x-LOW,HIGH-x)

def require_state(states,config):
    m=state_margins(states)
    if np.any(m<0):raise AssertionError('Negative raw state margin')
    if np.min(m/WIDTH)<=config['normalized_reserve']:raise AssertionError('Insufficient state numerical reserve')
    return m

def require_input(u):
    u=np.asarray(u,dtype=float).reshape(-1)
    if u.shape!=(2,) or not np.isfinite(u).all():raise ValueError('Nonfinite/invalid process command')
    m=np.minimum(u-ULO,UHI-u)
    if np.any(m<0):raise AssertionError('Negative raw requested-input margin')
    return m

def costs(states,commands,previous):
    x=np.asarray(states);u=np.asarray(commands);delta=np.diff(np.vstack([previous,u]),axis=0)
    tracking=(x[:-1,0]-.7)**2+(x[:-1,1]-.6)**2
    movement=.1*(delta[:,0]/100)**2+.001*(delta[:,1]/2000)**2
    terminal=float((x[-1,0]-.7)**2+(x[-1,1]-.6)**2)
    return {'tracking_sum':float(tracking.sum()),'movement_sum':float(movement.sum()),'terminal_cost':terminal,
        'stage_cost_sum':float((tracking+movement).sum()),'total_cost':float((tracking+movement).sum()+terminal),
        'return':-float((tracking+movement).sum()+terminal),
        'concentration_RMSE':np.sqrt(np.mean((x[:-1,:2]-[.7,.6])**2,axis=0)).tolist(),
        'feed_volume_L':float(10.01*DT*u[:,0].sum()),'cooling_energy_kJ':float(-DT*u[:,1].sum()),
        'physical_input_squared_sum':np.sum(u*u,axis=0).tolist(),'terminal_state':x[-1].tolist()}
