"""Directed-rounding comparison certificates for the real discrete Pendulum map.

This is not the floating-point Gymnasium transition implementation. Parameters
are exact rationals; initial states are the exact dyadics stored in the archive.
The checker in viability_checker.py does not import this module or MPFR.
"""

from fractions import Fraction
import math

import gmpy2 as g

MODEL = {
    'dt': '1/20', 'gravity': '10', 'mass': '1', 'length': '1',
    'theta_limit': '1', 'omega_limit': '3', 'torque_limit': '2',
    'speed_clip': '8',
    'arithmetic': 'real_rational_parameters_exact_binary_initials',
}
SCHEMA = 'pendulum_viability_extremal_v1'


def ratio(value):
    """Lossless dyadic serialization; never use a decimal display as evidence."""
    n, d = value.as_integer_ratio()
    return [str(n), str(d)]


def rational(pair):
    return Fraction(int(pair[0]), int(pair[1]))


def _endpoint(theta, omega, action, precision, rounding):
    # Division by positive exact integer 20 preserves lower/upper bounds even
    # for negative intermediates. A rounded literal 0.05 would not suffice.
    with g.context(precision=precision, round=rounding, trap_invalid=True,
                   trap_overflow=True, trap_underflow=True, trap_divzero=True):
        v = omega + (15 * g.sin(theta) + 3 * action) / 20
        t = theta + v / 20
        return t, v


def interval_step(box, action, precision):
    """Image enclosure from monotone endpoints; valid only from the safe box."""
    if action not in (-2, 2) or isinstance(action, bool):
        raise ValueError('Only the two declared constant extremal inputs are supported.')
    if precision not in (128, 256, 512):
        raise ValueError('Undeclared interval precision.')
    (tl, th), (wl, wh) = box
    if not (-1 <= tl <= th <= 1 and -3 <= wl <= wh <= 3):
        raise ValueError('Comparison domain violated; never propagate past the first exit.')
    lower = _endpoint(tl, wl, action, precision, g.RoundDown)
    upper = _endpoint(th, wh, action, precision, g.RoundUp)
    return ((lower[0], upper[0]), (lower[1], upper[1]))


def _row(step, box):
    return {'step': step, **{name: {'lo': ratio(pair[0]), 'hi': ratio(pair[1])}
                            for name, pair in zip(('theta', 'omega'), box)}}


def _safe(box):
    return all(-limit <= lo <= hi <= limit for (lo, hi), limit in zip(box, (1, 3)))


def _crossing(box, side):
    for name, (lo, hi), limit in zip(('theta', 'omega'), box, (1, 3)):
        if (side == 1 and lo > limit) or (side == -1 and hi < -limit):
            return name
    return None


def extremal_trajectory(state_id, initial_hex, action, precision=128, horizon=200):
    """Stop at the first exit/ambiguous boundary, preserving the safe prefix."""
    if type(state_id) is not int or len(initial_hex) != 2 or horizon != 200:
        raise ValueError('Expected an exact state ID, two hex coordinates, and H=200.')
    values = [float.fromhex(x) for x in initial_hex]
    if any(not math.isfinite(x) or x.hex() != text for x, text in zip(values, initial_hex)):
        raise ValueError('Noncanonical/nonfinite binary initial state.')
    if action not in (-2, 2) or isinstance(action, bool) or precision not in (128, 256, 512):
        raise ValueError('Undeclared action or precision.')
    with g.context(precision=precision, round=g.RoundToNearest):
        initial = [g.mpfr(v) for v in values]  # every binary64 value is exact here
    box = tuple((v, v) for v in initial)
    if not _safe(box):
        raise ValueError('Initial state is outside the closed safety set.')
    result = dict(schema=SCHEMA, state_id=state_id, initial_hex=list(initial_hex),
                  precision_bits=precision, action=action, horizon=horizon,
                  model=dict(MODEL), status='horizon_safe_extremal', decisive_step=None,
                  crossing_component=None, trace=[_row(0, box)])
    for step in range(1, horizon + 1):
        box = interval_step(box, action, precision)
        result['trace'].append(_row(step, box))
        if _safe(box):
            continue
        wanted = 1 if action == -2 else -1
        component = _crossing(box, wanted)
        if component:
            result.update(status='certified_nonviable', decisive_step=step,
                          crossing_component=component)
        else:
            component = _crossing(box, -wanted)
            result.update(status='opposite_exit' if component else 'boundary_unresolved',
                          decisive_step=step, crossing_component=component)
        break
    return result


def diagnostics(record):
    """Exact quantities plus explicitly approximate displays for the report."""
    widths = {name: max(rational(row[name]['hi']) - rational(row[name]['lo'])
                        for row in record['trace']) for name in ('theta', 'omega')}
    prefix = record['trace'] if record['status'] == 'horizon_safe_extremal' else record['trace'][:-1]
    prefix_margin = min(1 - max(abs(rational(row[name]['lo'])), abs(rational(row[name]['hi']))) / limit
                        for row in prefix for name, limit in [('theta', 1), ('omega', 3)])
    separation = None
    component = record['crossing_component']
    if component is not None:
        last = record['trace'][-1][component]
        lo, hi = rational(last['lo']), rational(last['hi'])
        limit = 1 if component == 'theta' else 3
        separation = lo - limit if lo > limit else -limit - hi
        assert separation > 0
    return {
        'interval_steps': len(record['trace']) - 1,
        'maximum_width_exact': {k: ratio(v) for k, v in widths.items()},
        'maximum_width_approx': {k: float(v) for k, v in widths.items()},
        'safe_prefix_minimum_margin_lower_exact': ratio(prefix_margin),
        'safe_prefix_minimum_margin_lower_approx': float(prefix_margin),
        'crossing_separation_lower_exact': None if separation is None else ratio(separation),
        'crossing_separation_lower_approx': None if separation is None else float(separation),
        'display_values_are_not_certificates': True,
        'scope': 'Declared real-valued discrete model only; no implementation-error envelope claimed.',
    }
