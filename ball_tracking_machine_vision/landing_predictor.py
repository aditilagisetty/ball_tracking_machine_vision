"""Extrapolating the filter state forward to find where the ball lands.

The filter tracks the ball's CENTER, so the ball lands when its center is
one radius above the floor (floor_height + ball_radius), the same rule
simulate_throw uses. The landing point reported is the spot on the floor
directly below the center at that moment (z = floor_height), which is what
the 10 cm floor grid measures.

Essentially, takes the velocities from the Kalman Filter to use for the landing 
prediction model. 
"""

import numpy as np

from ball_tracking_machine_vision.kalman_filter import (
    gravity_input,
    process_noise,
    transition_matrix,
)
from ball_tracking_machine_vision.simulation import BALL_RADIUS, GRAVITY


def propagate_trajectory(state, covariance, sigma_a, duration, dt=1 / 60):
    """Step a state and its uncertainty forward in time with no measurements

    Uses the filters own physics of F, gravity, and Q so the future is predicted
    exactly the way the filter predicts between frames. Works on copies so
    the live filter is never touched

    Returns dict with t (N), state (N, 6), covariance (N, 6, 6),
    starting at time 0 which is effectivelt now and ending exactly at duration
    """
    x = state.copy()
    P = covariance.copy()
    t = 0.0
    times, states, covariances = [t], [x.copy()], [P.copy()]

    while t < duration - 1e-12:          # tiny tolerance so rounding cant add an extra step
        step = min(dt, duration - t)     # last step is shorter to land exactly on duration
        F = transition_matrix(step)
        x = F @ x + gravity_input(step)                    # same as the filters predict
        P = F @ P @ F.T + process_noise(step, sigma_a)     # uncertainty grows each step
        t += step
        times.append(t)
        states.append(x.copy())
        covariances.append(P.copy())

    return {
        't': np.array(times),
        'state': np.array(states),
        'covariance': np.array(covariances),
    }


def find_landing_point(state, floor_height=0.0, ball_radius=BALL_RADIUS):
    """where and when the ball touches the floor assuming gravity only

    state: [x, y, z, vx, vy, vz] of the ball's center
    Contact is when the center reaches floor_height + ball_radius
    Returns (landing_point, time_to_landing) or None if it never lands.
    landing_point is on the floor directly below the center: (x, y, floor_height)
    """
    contact_height = floor_height + ball_radius
    x, y, z, vx, vy, vz = state

    # Solve z + vz*t - ½ g t² = contact_height for t using quadratic formula
    drop = z - contact_height
    discriminant = vz**2 + 2 * GRAVITY * drop
    if discriminant < 0:
        return None     # only possible if the ball is already below contact height

    t = (vz + np.sqrt(discriminant)) / GRAVITY   # the + root is the future one
    landing_point = np.array([x + vx * t, y + vy * t, floor_height])
    return landing_point, t


def predict_landing(kalman_filter, floor_height=0.0, ball_radius=BALL_RADIUS,
                    n_samples=500, rng=None):
    """Landing point and its uncertainty from the filter's current state.

    Contact is when the center reaches floor_height + ball_radius.

    Returns dict with
        'point': (3,) best-guess landing spot on the floor
        'time': seconds from now until landing
        'covariance': (2, 2) uncertainty of the landing (x, y), in m²
        'samples': (n, 3) the sampled landing points (for plotting the spread)
    or None if the ball can't land from this state.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Best guess: the filter's current state, landed with the exact formula.
    x, P = kalman_filter.get_state()  # copies: the live filter is not changed
    result = find_landing_point(x, floor_height, ball_radius)
    if result is None:
        return None
    point, time_to_landing = result

    # Carry the uncertainty forward to landing time. P grows on the way
    # covering pushes (drag, spin) the model doesn't know about
    future = propagate_trajectory(x, P, kalman_filter.sigma_a, time_to_landing)
    x_land = future['state'][-1]
    P_land = future['covariance'][-1]

    # Draw random states that are consistent with that uncertainty
    samples = rng.multivariate_normal(x_land, P_land, size=n_samples)

    # Land each one -- their spread is the landing uncertainty
    landings = []
    for sample in samples:
        sample_result = find_landing_point(sample, floor_height, ball_radius)
        if sample_result is not None:
            landings.append(sample_result[0])
    landings = np.array(landings)

    return {
        'point': point,
        'time': time_to_landing,
        'covariance': np.cov(landings[:, :2].T), # spread in x and y only bc z is the floor
        'samples': landings,
    }
