# src/safety.py

THETA_SAFE = 1.0
OMEGA_SAFE = 3.0


def safety_margin(theta, omega):
    """
    Normalized safety margin.

    margin > 0 : safe
    margin = 0 : on safety boundary
    margin < 0 : safety violation
    """

    theta_margin = (
        1.0 - abs(theta) / THETA_SAFE
    )

    omega_margin = (
        1.0 - abs(omega) / OMEGA_SAFE
    )

    return min(
        theta_margin,
        omega_margin
    )


def is_safe(theta, omega):
    """
    Returns True if both safety constraints hold.
    """

    return (
        abs(theta) <= THETA_SAFE
        and
        abs(omega) <= OMEGA_SAFE
    )