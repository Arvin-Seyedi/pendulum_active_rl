"""Independent, exact-rational checker for Experiment 08 extremal certificates.

This module does not import the MPFR interval producer, NumPy, or Gymnasium.
It bounds sine by consecutive alternating Taylor sums using Fraction arithmetic.
On [-1, 1], the terms decrease in magnitude, so these are rigorous enclosures.

The checked theorem concerns the real-valued discrete map with rational physical
parameters and exact binary initial values. It is not a certificate about libm,
floating-point Gymnasium execution, continuous-time motion, or a physical plant.

Proof facts used here: on the safe strip, cos(theta) >= 1-theta**2/2 >= 1/2,
so every state/action derivative of the discrete map is nonnegative. Starting
in S, |omega_next| <= 3+(15+6)/20 = 81/20 < 8; speed clipping is inactive.
Thus a constant -2 path with a safe prefix and a strict upper crossing bounds
every hypothetical safe admissible path from below. Constant +2 is symmetric.
"""

from fractions import Fraction
from functools import lru_cache
import math


MODEL = {
    'dt': '1/20', 'gravity': '10', 'mass': '1', 'length': '1',
    'theta_limit': '1', 'omega_limit': '3', 'torque_limit': '2',
    'speed_clip': '8',
    'arithmetic': 'real_rational_parameters_exact_binary_initials',
}
SCHEMA = 'pendulum_viability_extremal_v1'
STATUSES = {'certified_nonviable', 'opposite_exit',
            'horizon_safe_extremal', 'boundary_unresolved'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rational(pair):
    _require(isinstance(pair, list) and len(pair) == 2
             and all(isinstance(v, str) for v in pair),
             'Interval endpoints must be numerator/denominator string pairs.')
    try:
        numerator, denominator = map(int, pair)
    except ValueError as exc:
        raise ValueError('Invalid rational endpoint.') from exc
    _require(denominator > 0, 'Rational denominators must be positive.')
    return Fraction(numerator, denominator)


def _interval(item):
    _require(isinstance(item, dict) and set(item) == {'lo', 'hi'},
             'Malformed interval.')
    lower, upper = _rational(item['lo']), _rational(item['hi'])
    _require(lower <= upper, 'Reversed interval.')
    return lower, upper


def _inside(box):
    theta, omega = box
    return -1 <= theta[0] <= theta[1] <= 1 and -3 <= omega[0] <= omega[1] <= 3


@lru_cache(maxsize=4096)
def sine_bounds(x, terms=24):
    """Exact rational enclosure of sin(x), valid only for |x| <= 1.

For x >= 0, consecutive partial sums bound sin(x) by the alternating-series
theorem. Odd symmetry handles negative x without changing the theorem's domain.
The omitted term is not estimated by floating-point arithmetic.
"""
    x = Fraction(x)
    _require(abs(x) <= 1, 'Sine checker domain is the safe angle strip.')
    _require(type(terms) is int and terms >= 1, 'Positive sine term count required.')
    if x == 0:
        return Fraction(0), Fraction(0)
    if x < 0:
        lower, upper = sine_bounds(-x, terms)
        return -upper, -lower
    term, total = x, x
    for k in range(1, terms):
        term *= -x*x / ((2*k)*(2*k+1))
        total += term
    next_term = term * (-x*x) / ((2*terms)*(2*terms+1))
    adjacent = total + next_term
    return min(total, adjacent), max(total, adjacent)


def _image(box, action, terms):
    """Independent monotone endpoint image; no producer arithmetic is reused."""
    theta, omega = box
    sine_lower = sine_bounds(theta[0], terms)[0]
    sine_upper = sine_bounds(theta[1], terms)[1]
    velocity_lower = omega[0] + Fraction(3, 4)*sine_lower + Fraction(3, 20)*action
    velocity_upper = omega[1] + Fraction(3, 4)*sine_upper + Fraction(3, 20)*action
    return ((theta[0] + velocity_lower/20, theta[1] + velocity_upper/20),
            (velocity_lower, velocity_upper))


def _contains(container, enclosed):
    return all(outer[0] <= inner[0] <= inner[1] <= outer[1]
               for outer, inner in zip(container, enclosed))


def _crossings(box):
    upper = {'theta': box[0][0] > 1, 'omega': box[1][0] > 3}
    lower = {'theta': box[0][1] < -1, 'omega': box[1][1] < -3}
    return upper, lower


def check_extremal_certificate(record, expected_state_id=None,
                               expected_initial_hex=None):
    """Check a serialized enclosure and its claimed stopping status.

Raises ValueError for malformed records or claims not proved by exact rational
checks. A valid noncertifying record returns passed=True, certified=False;
passing the checker must not be confused with proving nonviability.
"""
    _require(isinstance(record, dict), 'Certificate must be a dictionary.')
    _require(record.get('schema') == SCHEMA, 'Unexpected certificate schema.')
    _require(record.get('model') == MODEL, 'Unapproved mathematical model.')
    state_id = record.get('state_id')
    _require(type(state_id) is int and state_id >= 0, 'Invalid state ID.')
    if expected_state_id is not None:
        _require(state_id == expected_state_id, 'State ID does not match the source manifest.')
    precision = record.get('precision_bits')
    _require(type(precision) is int and precision in (128, 256, 512),
             'Unexpected declared precision.')
    action = record.get('action')
    _require(type(action) is int and action in (-2, 2), 'Extremal action must be -2 or +2.')
    horizon = record.get('horizon')
    _require(type(horizon) is int and horizon == 200, 'The approved horizon is 200 steps.')
    status = record.get('status')
    _require(status in STATUSES, 'Unknown stopping status.')
    initial_hex = record.get('initial_hex')
    _require(isinstance(initial_hex, list) and len(initial_hex) == 2
             and all(isinstance(v, str) for v in initial_hex), 'Two hexadecimal initial values required.')
    try:
        initial_float = [float.fromhex(v) for v in initial_hex]
    except ValueError as exc:
        raise ValueError('Invalid hexadecimal initial value.') from exc
    _require(all(math.isfinite(v) for v in initial_float), 'Initial values must be finite.')
    _require([v.hex() for v in initial_float] == initial_hex,
             'Initial hexadecimal values must be canonical, preserving signed zero.')
    if expected_initial_hex is not None:
        _require(initial_hex == list(expected_initial_hex),
                 'Initial binary values do not match the source manifest.')
    initial = [Fraction(*v.as_integer_ratio()) for v in initial_float]
    trace = record.get('trace')
    _require(isinstance(trace, list) and 2 <= len(trace) <= horizon+1,
             'Trace must contain initial state and at least one transition, within horizon.')
    boxes = []
    for index, item in enumerate(trace):
        _require(isinstance(item, dict) and type(item.get('step')) is int
                 and item['step'] == index, 'Trace indices must be consecutive from zero.')
        _require('theta' in item and 'omega' in item, 'Both state components required.')
        boxes.append((_interval(item['theta']), _interval(item['omega'])))
    _require(boxes[0] == ((initial[0], initial[0]), (initial[1], initial[1])),
             'Initial enclosure must equal the exact binary initial point.')
    _require(_inside(boxes[0]), 'Initial state is outside the safety set.')
    base_terms = max(24, (precision+7)//8)
    terms_used = []
    for k, (previous, following) in enumerate(zip(boxes, boxes[1:]), start=1):
        _require(_inside(previous), f'Unsafe or ambiguous prefix enclosure at step {k-1}.')
        # A narrower mathematical sine enclosure may be necessary to validate a
        # very tight published MPFR interval. Increasing terms changes no model
        # or tolerance and never repairs an enclosure by widening its endpoints.
        checked = False
        for terms in (base_terms, 2*base_terms):
            if _contains(following, _image(previous, action, terms)):
                checked = True
                terms_used.append(terms)
                break
        _require(checked, f'Published enclosure does not contain the independent image at step {k}.')
    last_step = len(boxes)-1
    upper, lower = _crossings(boxes[-1])
    intended, opposite = (upper, lower) if action == -2 else (lower, upper)
    component = record.get('crossing_component')
    decisive = record.get('decisive_step')
    if status == 'certified_nonviable':
        _require(decisive == last_step and type(decisive) is int,
                 'Decisive crossing step must equal the final trace index.')
        _require(component in ('theta', 'omega') and intended[component],
                 'Claimed nonviability lacks a strict whole-interval crossing in the correct direction.')
    elif status == 'opposite_exit':
        _require(decisive == last_step and type(decisive) is int,
                 'Opposite exit step must equal the final trace index.')
        _require(component in ('theta', 'omega') and opposite[component]
                 and not any(intended.values()),
                 'Opposite exit must be strict and cannot be an intended crossing.')
    elif status == 'horizon_safe_extremal':
        _require(last_step == horizon and _inside(boxes[-1])
                 and decisive is None and component is None,
                 'Safe-horizon status requires all 201 enclosures inside S.')
    else:
        _require(not _inside(boxes[-1]) and not any(upper.values()) and not any(lower.values()),
                 'Unresolved boundary must overlap a safety boundary without a strict crossing.')
        _require(decisive is None or type(decisive) is int and decisive == last_step,
                 'Unresolved step must be absent or equal to the final index.')
        _require(component is None or component in ('theta', 'omega'),
                 'Unknown unresolved component.')
    return {
        'passed': True,
        'check_count': last_step,
        'state_id': state_id,
        'action': action,
        'status': status,
        'certified': status == 'certified_nonviable',
        'model_only': True,
        'independent_arithmetic': 'stdlib Fraction; rational alternating Taylor sine enclosures',
        'sine_terms_min': min(terms_used),
        'sine_terms_max': max(terms_used),
        'prefix_enclosures_inside_safety_set': True,
        'all_transition_enclosures_verified': True,
        'comparison_direction': 'lower_path_upper_exit' if action == -2 else 'upper_path_lower_exit',
        'speed_clipping_inactive_bound': '81/20 < 8',
        'numerical_tolerance': '0',
    }
