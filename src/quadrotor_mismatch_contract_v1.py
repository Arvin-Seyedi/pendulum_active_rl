"""Prospective E12_MMAQ_v1 discrepancy bounds; no environment/controller calls.

All predictions restart at the supplied *current* state and cache. They are
matched-state checks, not additional control trajectories or a plant replacement.
The proof covers the declared real source recurrence. Its 1e-10 correspondence
to native binary64 PyBullet is an empirical numerical contract, not a theorem
about arbitrary Bullet executions, contacts or nonplanar motion.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
import itertools
import numpy as np


def draw_states(seed, count, low, high, excluded=()):
    """Native PCG64 doubles, row-major. Only coordinate duplicates are rejected."""
    generator = np.random.Generator(np.random.PCG64(seed))
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    forbidden = {np.asarray(x, dtype='<f8').tobytes() for x in excluded}
    states, attempts = [], []
    while len(states) < count:
        unit = generator.random(6)
        x = low + (high-low)*unit
        key = np.asarray(x, dtype='<f8').tobytes()
        attempts.append({'draw': len(attempts), 'unit_hex': [float(v).hex() for v in unit],
                         'accepted': key not in forbidden})
        if key not in forbidden:
            forbidden.add(key); states.append(x)
    return np.asarray(states), attempts


def state_manifest(states, attempts, seed):
    return {'seed': seed, 'bit_generator': 'PCG64', 'draws': attempts,
            'states': [{'id': i, 'coordinates': x.tolist(),
                        'binary64_hex': [float(v).hex() for v in x],
                        'little_endian_bytes_hex': np.asarray(x, dtype='<f8').tobytes().hex()}
                       for i, x in enumerate(states)]}


def action_menu(f):
    width = np.asarray(f['action_high'])-f['action_low']
    return [(width*np.asarray(pair)).tolist()
            for pair in itertools.product([-.05, 0., .05], repeat=2)]


def executed_thrust(action, f):
    """Independent native two-group PWM/RPM mapping, known before env.step."""
    u = np.clip(np.asarray(action, dtype=float), f['action_low'], f['action_high'])
    pwm = (np.sqrt(np.maximum(u, 0)/2/f['KF'])-f['PWM2RPM_CONST'])/f['PWM2RPM_SCALE']
    pwm = np.clip(np.r_[pwm, pwm[::-1]], f['MIN_PWM'], f['MAX_PWM'])
    rpm = f['PWM2RPM_SCALE']*pwm+f['PWM2RPM_CONST']
    actual = f['KF']*np.array([rpm[0]**2+rpm[3]**2, rpm[1]**2+rpm[2]**2])
    return u, actual, rpm


def linear_prediction(x, u, f, dt=.02):
    """Exact finite polynomial for the frozen hover-linearized ZOH model."""
    x, u = np.asarray(x), np.asarray(u)
    alpha = f['arm_length']/np.sqrt(2)/f['Iyy']*(u[1]-u[0])
    hover_accel = sum(f['hover_input'])/f['mass']
    y = x.copy()
    y[0] += dt*x[1]+hover_accel*(x[4]*dt**2/2+x[5]*dt**3/6+alpha*dt**4/24)
    y[1] += hover_accel*(x[4]*dt+x[5]*dt**2/2+alpha*dt**3/6)
    az = (sum(u)-sum(f['hover_input']))/f['mass']
    y[2] += dt*x[3]+az*dt**2/2; y[3] += az*dt
    y[4] += dt*x[5]+alpha*dt**2/2; y[5] += alpha*dt
    return y


def bound_components(x, requested, cache, f, linear=False, frequency=1000):
    """Analytic majorants, evaluated upwards; never consumes successor states.

    alpha_s/p: symbolic/URDF angular acceleration for requested/executed thrust.
    Omega bounds source angular rates; delta bounds the cache angle difference.
    A bounds angular acceleration mismatch. Etheta includes right-rectangle
    semi-implicit integration. Translational left-rectangle errors use a global
    derivative bound over this one control interval. RK4 stage-angle error and
    Simpson fourth-derivative remainder bound its difference to the symbolic ODE.
    Linear MPC uses its exact polynomial plus a nonlinearization remainder.
    """
    x, requested = np.asarray(x), np.asarray(requested)
    if not np.isfinite(np.r_[x, requested, cache]).all():
        raise ValueError('Nonfinite contract input')
    if frequency not in (1000, 2000): raise ValueError('Unapproved frequency')
    model_u, actual_u, rpm = executed_thrust(requested, f)
    def q(v): return Fraction(float(v))
    als = q(f['arm_length']/np.sqrt(2))*(q(model_u[1])-q(model_u[0]))/q(f['Iyy'])
    alp = q(.028)*(q(actual_u[1])-q(actual_u[0]))/q(f['Iyy'])
    bs = (q(model_u[0])+q(model_u[1]))/q(f['mass'])
    bp = (q(actual_u[0])+q(actual_u[1]))/q(f['mass'])
    with localcontext() as ctx:
        ctx.prec = 80; ctx.rounding = ROUND_CEILING
        def d(v):
            v = v if isinstance(v, Fraction) else q(v)
            return Decimal(v.numerator)/Decimal(v.denominator)
        D = Decimal(1)/50; h = Decimal(1)/frequency
        a_s, a_p = d(abs(als)), d(abs(alp))
        beta_s, beta_p = d(bs), d(bp)
        omega_s = d(abs(q(x[5])))+D*a_s
        omega_p = d(abs(q(x[5])))+D*a_p
        delta = max(d(abs(q(x[4])-q(cache))), h*omega_p)
        geometry_conversion = d(abs(alp-als))
        cache_accel = a_p*delta**2/2
        A = geometry_conversion+cache_accel
        etheta = A*D*(D+h)/2+a_s*D*h/2
        eomega = D*A
        derivative = beta_s*omega_s
        accel_error = d(abs(bp-bs))+beta_p*(delta+etheta)
        ev = D*accel_error+derivative*D*h/2
        ep_x = D*ev+beta_s*D*h/2
        ep_z = D*ev+(beta_s+d(f['gravity']))*D*h/2
        base = [ep_x, ev, ep_z, ev, etheta, eomega]
        rk4 = [Decimal(0)]*6; lin = [Decimal(0)]*6
        if linear:
            theta = d(abs(q(x[4])))+d(abs(q(x[5])))*D+a_s*D**2/2
            g_lin = (q(f['hover_input'][0])+q(f['hover_input'][1]))/q(f['mass'])
            ax = beta_s*theta**3/6+d(abs(bs-g_lin))*theta
            az = beta_s*theta**2/2+d(abs(g_lin-q(f['gravity'])))
            lin = [ax*D**2/2, ax*D, az*D**2/2, az*D, Decimal(0), Decimal(0)]
        else:
            fourth = omega_s**4+6*omega_s**2*a_s+3*a_s**2
            rv = beta_s*a_s*D**3/12+beta_s*fourth*D**5/2880
            rp = derivative*D**3/6+beta_s*D**2/3*(d(abs(q(x[5])))*D/2+a_s*D**2/4)
            rk4 = [rp, rv, rp, rv, Decimal(0), Decimal(0)]
        def outward(values):
            return np.array([np.nextafter(float(v), np.inf) for v in values])
        total = [a+b+c for a,b,c in zip(base, rk4, lin)]
        return {'bound': outward(total), 'source_and_integration': outward(base),
                'rk4': outward(rk4), 'linearization': outward(lin),
                'cache_angle_bound': float(delta), 'rate_bound': float(omega_p),
                'executed_thrust': actual_u, 'model_thrust': model_u, 'rpm': rpm,
                'angular_geometry_conversion_bound': float(D*geometry_conversion),
                'angular_cache_bound': float(D*cache_accel)}


def source_prediction(x, actual_u, cache, f, frequency=1000):
    """Independent scalar transcription; one matched-state interval only."""
    y = np.asarray(x, dtype=float).copy(); c = float(cache); h = 1/frequency
    total = sum(actual_u)/f['mass']; alpha = .028/f['Iyy']*(actual_u[1]-actual_u[0])
    for _ in range(frequency//50):
        theta = y[4]
        y[1] += h*total*np.sin(c); y[3] += h*(total*np.cos(c)-f['gravity'])
        y[5] += h*alpha*np.cos(theta-c)
        y[[0,2,4]] += h*y[[1,3,5]]
        c = theta
    return y
