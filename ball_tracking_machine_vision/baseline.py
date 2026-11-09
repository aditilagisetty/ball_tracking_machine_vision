"""polynomial fit baseline to compare against the Kalman filter

fit curves to the raw 3D measurements
seen so far and extend them to the floor 
No filter, no noise model, and no
knowledge of gravity
If the Kalman filter can't beat this then its now worth it!

    x(t) = a_x·t + b_x            straight line (no force sideways)
    y(t) = a_y·t + b_y            straight line
    z(t) = a_z·t² + b_z·t + c_z   parabola (gravity bends it w strength unknown)
"""

import numpy as np

from ball_tracking_machine_vision.simulation import BALL_RADIUS


def fit_trajectory(timestamps, positions):
    """Fit a line to x and y and a parabola to z, using every measurement so far

    Needs at least 3 measurements for the parabola

    Returns dict with each axis' polynomial coefficients and the time of the last measurement
    """
    t = np.asarray(timestamps, dtype=float)
    p = np.asarray(positions, dtype=float)
    if len(t) < 3:
        raise ValueError('need at least 3 measurements to fit a parabola')

    return {
        'x': np.polyfit(t, p[:, 0], 1),
        'y': np.polyfit(t, p[:, 1], 1),
        'z': np.polyfit(t, p[:, 2], 2),
        't_last': t[-1],
    }


def predict_landing_baseline(fit, floor_height=0.0, ball_radius=BALL_RADIUS):
    """Landing point from the polynomial fit

    contact is when the center reaches
    floor_height + ball_radius, and the landing point is on the floor below it

    Returns (landing_point, landing_time) with landing_time on the same clock
    as the timestamps that were fitted or returns None if the fitted curve never comes
    back down
    """
    contact_height = floor_height + ball_radius

    # Solve z(t) = contact_height
    #subtract it from the constant term and find roots
    z_coefficients = fit['z'].copy()
    z_coefficients[-1] -= contact_height
    roots = np.roots(z_coefficients)

    # Keep real roots in the future; the landing is the first of those.
    future = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > fit['t_last']]
    if not future:
        return None
    t_land = min(future)

    landing_point = np.array([
        np.polyval(fit['x'], t_land),
        np.polyval(fit['y'], t_land),
        floor_height,
    ])
    return landing_point, t_land
