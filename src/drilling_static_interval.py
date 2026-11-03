"""E11B static interval FEM arithmetic; never advances a state or calls a controller.

Inputs are exact dyadics copied from native workspaces, not regenerated decimal
parameters. Every elementary operation is enclosed with MPFR directed rounding.
The two allowed precisions are part of the approved diagnostic protocol.
"""
import gmpy2 as g


class Interval:
    def __init__(self, value=0, precision=128, upper=None):
        if precision not in (128, 256):
            raise ValueError('Undeclared precision')
        self.p = precision
        with g.context(precision=precision, round=g.RoundDown):
            self.lo = g.mpfr(value)
        with g.context(precision=precision, round=g.RoundUp):
            self.hi = g.mpfr(value if upper is None else upper)
        if not self.lo <= self.hi:
            raise ValueError('Invalid interval')

    def coerce(self, x):
        if isinstance(x, Interval):
            if x.p != self.p:
                raise ValueError('Mixed precision')
            return x
        return Interval(x, self.p)

    def __add__(self, x):
        x = self.coerce(x)
        with g.context(precision=self.p, round=g.RoundDown):
            lo = self.lo + x.lo
        with g.context(precision=self.p, round=g.RoundUp):
            hi = self.hi + x.hi
        return Interval(lo, self.p, hi)

    __radd__ = __add__

    def __neg__(self):
        with g.context(precision=self.p):
            return Interval(-self.hi, self.p, -self.lo)

    def __sub__(self, x):
        return self + -self.coerce(x)

    def __rsub__(self, x):
        return self.coerce(x) - self

    def __mul__(self, x):
        x = self.coerce(x)
        with g.context(precision=self.p, round=g.RoundDown):
            lo = min(a*b for a in (self.lo, self.hi) for b in (x.lo, x.hi))
        with g.context(precision=self.p, round=g.RoundUp):
            hi = max(a*b for a in (self.lo, self.hi) for b in (x.lo, x.hi))
        return Interval(lo, self.p, hi)

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = self.coerce(x)
        if x.lo <= 0 <= x.hi:
            raise ArithmeticError('Unresolved zero denominator')
        with g.context(precision=self.p, round=g.RoundDown):
            lo = 1 / x.hi
        with g.context(precision=self.p, round=g.RoundUp):
            hi = 1 / x.lo
        return self * Interval(lo, self.p, hi)

    def __rtruediv__(self, x):
        return self.coerce(x) / self

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError('Only nonnegative integer powers')
        result = self.coerce(1)
        for _ in range(n):
            result = result * self
        return result

    def function(self, name):
        fun = getattr(g, name)
        loarg, hiarg = self.lo, self.hi
        if name == 'sqrt' and self.lo < 0:
            raise ArithmeticError('Unresolved square-root domain')
        if name in ('sin', 'cos', 'tan'):
            if not -1 <= self.lo <= self.hi <= 1:
                raise ArithmeticError('Trigonometric monotonicity domain exceeded')
            if name == 'cos':
                with g.context(precision=self.p):
                    loarg = max(abs(self.lo), abs(self.hi))
                    hiarg = 0 if self.lo <= 0 <= self.hi else min(abs(self.lo), abs(self.hi))
        with g.context(precision=self.p, round=g.RoundDown):
            lo = fun(loarg)
        with g.context(precision=self.p, round=g.RoundUp):
            hi = fun(hiarg)
        return Interval(lo, self.p, hi)

    def midpoint(self):
        with g.context(precision=self.p, round=g.RoundToNearest):
            return (self.lo+self.hi)/2

    def record(self):
        def ratio(x):
            n, d = x.as_integer_ratio()
            return [str(n), str(d)]
        with g.context(precision=self.p, round=g.RoundUp):
            width = self.hi-self.lo
        return {'lo': ratio(self.lo), 'hi': ratio(self.hi),
                'midpoint_display': float(self.midpoint()), 'width_display': float(width),
                'excludes_zero': bool(self.lo > 0 or self.hi < 0)}


def zeros(n, m, p):
    return [[Interval(0, p) for _ in range(m)] for _ in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    if len(a[0]) != len(b):
        raise ValueError('Matrix shape mismatch')
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def solve(a, b):
    """Interval Gaussian elimination with fixed midpoint partial pivoting.

    A nonzero interval pivot is required. Interval operations enclose the
    point-matrix solve from the declared primitive inputs; no floating inverse
    or unverified approximate-solution assertion is used.
    """
    n = len(a)
    x = [list(row)+list(rhs) for row, rhs in zip(a, b)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(x[i][k].midpoint()))
        x[k], x[pivot] = x[pivot], x[k]
        if x[k][k].lo <= 0 <= x[k][k].hi:
            raise ArithmeticError('Unresolved interval pivot')
        for i in range(k+1, n):
            factor = x[i][k]/x[k][k]
            for j in range(k+1, len(x[i])):
                x[i][j] = x[i][j]-factor*x[k][j]
            x[i][k] = Interval(0, x[i][k].p)
    out = zeros(n, len(b[0]), a[0][0].p)
    for i in reversed(range(n)):
        for j in range(len(b[0])):
            out[i][j] = (x[i][n+j]-sum(x[i][k]*out[k][j] for k in range(i+1, n)))/x[i][i]
    return out


def cross(x, y):
    return [x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]]


def unit(v):
    length = sum(x*x for x in v).function('sqrt')
    return [x/length for x in v]


def reassemble_intervals(a, precision):
    """Independent interval assembly from frozen primitive binary64 data.

    Saved M/K/D/R/reduced matrices are deliberately not inputs to assembly.
    Geometry coordinates, angles and section/material scalars are frozen at
    their archived values. This is not a claim about exact engineering inputs.
    """
    p = precision
    scalar = lambda k: Interval(float(a[k].item()), p)
    pos = [[Interval(float(x), p) for x in row[:3]] for row in a['P_initial'][0]]
    tail = [Interval(float(x), p) for x in a['P_tail'].reshape(-1)[:3]]
    angles = [Interval(float(x), p) for x in a['angle'].reshape(-1)]
    m, l, young, shear, area, ix, iy, iz, polar = [scalar(k) for k in ('m','l','E','G','A','Ix','Iy','Iz','J')]
    mass, stiff, damp, gravity = zeros(20,20,p), zeros(20,20,p), zeros(20,20,p), zeros(20,1,p)
    rotations = []
    for i in range(4):
        d = [pos[i+1][j]-pos[i][j] for j in range(3)]
        chord = [(pos[i+2] if i < 3 else tail)[j]-pos[i][j] for j in range(3)]
        tangent, normal = unit(d), unit(cross(d, chord))
        lateral = cross(normal, tangent)
        r = [tangent[:2], lateral[:2]]
        rot = zeros(8,8,p)
        for block in range(4):
            for j in range(2):
                for k in range(2): rot[2*block+j][2*block+k] = r[j][k]
        rotations.append(rot)
        km, mm = zeros(8,8,p), zeros(8,8,p)
        for ids, factor, coeff in [([0,4],young*area/l,[[1,-1],[-1,1]]),
                ([1,5],12*young*iz/l**3,[[1,-1],[-1,1]]),
                ([2,6],shear*polar/l,[[1,-1],[-1,1]]),
                ([3,7],young*iy/l,[[4,2],[2,4]])]:
            for j in range(2):
                for k in range(2): km[ids[j]][ids[k]] = factor*coeff[j][k]
        for ids, factor, coeff in [([0,4],m/420,[[140,70],[70,140]]),
                ([1,5],m/420,[[156,54],[54,156]]),
                ([2,6],m*ix/area/420,[[140,-70],[-70,140]]),
                ([3,7],m*l*l/420,[[4,-3],[-3,4]])]:
            for j in range(2):
                for k in range(2): mm[ids[j]][ids[k]] = factor*coeff[j][k]
        kg = multiply(transpose(rot),multiply(km,rot))
        mg = multiply(transpose(rot),multiply(mm,rot))
        fg = zeros(8,1,p)
        for j in range(2):
            th = angles[i+j]
            fg[4*j][0] = (th.function('cos')-th.function('sin')*scalar('fri_coef'))*m*scalar('g')/2
        fg = multiply(transpose(rot),fg)
        for j in range(8):
            gravity[4*i+j][0] += fg[j][0]
            for k in range(8):
                mass[4*i+j][4*i+k] += mg[j][k]
                stiff[4*i+j][4*i+k] += kg[j][k]
                damp[4*i+j][4*i+k] += scalar('Raleigh_a')*mg[j][k]+scalar('Raleigh_b')*kg[j][k]
    transform = zeros(20,10,p)
    for i in range(5):
        for offset in (0,2):
            transform[4*i+offset][2*i+offset//2] = Interval(1,p)
            transform[4*i+offset+1][2*i+offset//2] = angles[i].function('tan')
    # Force signs and terminal nodal frame reconstructed independently.
    nominal = [list(row) for row in gravity]
    nominal[0][0] += scalar('force_d'); nominal[2][0] += scalar('torque_d')
    c, s = angles[-1].function('cos'), angles[-1].function('sin')
    nominal[16][0] -= c*scalar('WOB_d'); nominal[17][0] -= s*scalar('WOB_d')
    nominal[18][0] -= c*scalar('TOB_d'); nominal[19][0] -= s*scalar('TOB_d')
    rhs = [stiff[i]+damp[i]+nominal[i] for i in range(20)]
    solved = solve(mass,rhs)
    km = multiply([row[:20] for row in solved[::2]],transform)
    dm = multiply([row[20:40] for row in solved[::2]],transform)
    fm = [[row[40]] for row in solved[::2]]
    vs = [[Interval(float(x),p)] for x in a['Vs'].reshape(-1)]
    ic = [[Interval(float(x),p)] for x in a['IC'].reshape(-1)]
    kv, dv, ki = multiply(km,vs), multiply(dm,vs), multiply(km,ic)
    r0 = [[fm[i][0]-dv[i][0]-ki[i][0]] for i in range(10)]
    return dict(M=mass,K=stiff,D=damp,KM=km,DM=dm,FM=fm,T=transform,
                gravity=gravity,r0=r0,KM_Vs=kv,rotations=rotations)


def nominal_load_intervals(a, precision, native_pi=False):
    p = precision
    s = lambda k: Interval(float(a[k].item()),p)
    c = Interval(float(a['angle'].reshape(-1)[-1]),p).function('cos')
    v = Interval(float(a['Vs'].reshape(-1)[8]),p)/c
    w = Interval(float(a['Vs'].reshape(-1)[9]),p)/c
    with g.context(precision=p,round=g.RoundDown): lo = g.const_pi()
    with g.context(precision=p,round=g.RoundUp): hi = g.const_pi()
    # Distinguish mathematical pi from the native binary64 pi constant.
    import math
    pi = Interval(math.pi,p) if native_pi else Interval(lo,p,hi)
    d = 2*pi*v/w
    if not d.lo > s('d0').hi or not d.hi < s('threshold').lo:
        raise ArithmeticError('Nominal load branch unresolved')
    fv = 1/(1+(s('a1')*v+s('b1')).function('exp'))
    fw = (s('a2')*w+s('b2')).function('tanh')
    wc = s('epsilon')*s('zeta')*s('Rb')*d
    tc = s('epsilon')*s('Rb')**2*d/2
    wf = s('Rb')*s('sigma')*s('lw')*fv
    tf = s('gam')*s('miu')*s('Rb')*wf*fw/2
    return {'v':v,'w':w,'d':d,'WOB':wc+wf,'TOB':tc+tf}
