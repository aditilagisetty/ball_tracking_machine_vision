"""Extrapolating the filter state forward to find where the ball lands.

The filter tracks the ball's CENTER, so the ball lands when its center is
one radius above the floor (floor_height + ball_radius), the same rule
simulate_throw uses. The landing point reported is the spot on the floor
directly below the center at that moment (z = floor_height), which is what
the 10 cm floor grid measures.

essentially takes the velocities from the Kalman Filter to use for the landing
prediction model

the filter itself is gravity only (sigma_a covers drag frame to frame)
but over a whole flight our light ball slows down a lot, so a gravity only
guess lands too far. So the landing prediction flies the ball forward with
drag acceleration = gravity - drag * |v| * v, the same physics as the
simulation. drag=0 gives back the exact gravity only formula.
"""

import numpy as np

from ball_tracking_machine_vision.kalman_filter import (
    gravity_input,
    process_noise,
    transition_matrix,
)
from ball_tracking_machine_vision.simulation import BALL_DRAG, BALL_RADIUS, GRAVITY

# Drag has no exact formula so we take many tiny steps instead
FLIGHT_STEP = 1 / 1000

GRAVITY_VECTOR = np.array([0.0, 0.0, -GRAVITY])

# No hand throw gets near this (m/s); see fly_to_floor.
MAX_SPEED = 50.0


def acceleration(velocity, drag):
    """Gravity plus drag for one or many velocities """
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    return GRAVITY_VECTOR - drag * speed * velocity


def drag_jacobian(velocity, drag):
    """How the drag acceleration changes when the velocity changes a little (3×3)

    Drag is -k|v|v, and its derivative is -k (|v| I + v vᵀ / |v|).
    Needed to carry the uncertainty P forward when drag is on the same way
    F carries it forward for the straight line part
    """
    speed = np.linalg.norm(velocity)
    if speed < 1e-9:
        return np.zeros((3, 3))
    return -drag * (speed * np.eye(3) + np.outer(velocity, velocity) / speed)


def propagate_trajectory(state, covariance, sigma_a, duration, dt=1 / 60, drag=0.0):
    """Step a state and its uncertainty forward in time with no measurements

    Uses the filters own physics of F, gravity, and Q so the future is predicted
    exactly the way the filter predicts between frames. Works on copies so
    the live filter is never touched

    With drag > 0 the state is flown forward with drag
    and P is carried forward with F linearized around the current velocity
    Q is still added every step.

    Returns dict with t (N), state (N, 6), covariance (N, 6, 6),
    starting at time 0 which is effectivelt now and ending exactly at duration
    """
    x = state.copy()
    P = covariance.copy()
    t = 0.0
    times, states, covariances = [t], [x.copy()], [P.copy()]

    while t < duration - 1e-12:          # tiny tolerance so rounding cant add an extra step
        step = min(dt, duration - t)     # last step is shorter to land exactly on duration

        if drag == 0:
            F = transition_matrix(step)
            x = F @ x + gravity_input(step)                # same as the filters predict
        else:
            # F for this step position follows velocity and velocity
            # changes with drag. A is the rate of change matrix, F ≈ I + A dt + A² dt²/2
            A = np.zeros((6, 6))
            A[:3, 3:] = np.eye(3)
            A[3:, 3:] = drag_jacobian(x[3:], drag)
            F = np.eye(6) + A * step + A @ A * step**2 / 2

            # the state itself is flown forward exactlyish with small substeps
            n_sub = max(1, int(np.ceil(step / FLIGHT_STEP)))
            h = step / n_sub
            for _ in range(n_sub):
                a = acceleration(x[3:], drag)
                x[:3] = x[:3] + x[3:] * h + 0.5 * a * h**2
                x[3:] = x[3:] + a * h

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


def find_landing_point(state, floor_height=0.0, ball_radius=BALL_RADIUS, drag=0.0):
    """where and when the ball touches the floor

    state: [x, y, z, vx, vy, vz] of the ball's center
    drag: 0 for gravity only (exact formula), or BALL_DRAG for our ball
    Contact is when the center reaches floor_height + ball_radius
    Returns (landing_point, time_to_landing) or None if it never lands.
    landing_point is on the floor directly below the center: (x, y, floor_height)
    time_to_landing can be negative if the ball is already past contact height
    """
    if drag != 0:
        points, times = fly_to_floor(np.atleast_2d(state), floor_height, ball_radius, drag)
        if np.isnan(times[0]):
            return None
        return points[0], times[0]

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


def fly_to_floor(states, floor_height=0.0, ball_radius=BALL_RADIUS, drag=BALL_DRAG,
                 step=FLIGHT_STEP, max_time=10.0):
    """fly many balls with gravity + drag until each one reaches contact height

    states: (N, 6) starting states
    All N are stepped together 
    Balls above contact height fly forward until they come down through it
    Balls already below it and falling are flown BACKWARD in time until they
    were at contact height. 

    Returns (points (N, 3), times (N,)). Balls that never reach contact
    height get NaN
    """
    states = np.asarray(states, dtype=float)
    position = states[:, :3].copy()
    velocity = states[:, 3:].copy()
    n = len(states)
    contact_height = floor_height + ball_radius

    # +1 = fly forward, -1 = fly backward
    below_and_falling = (position[:, 2] <= contact_height) & (velocity[:, 2] <= 0)
    direction = np.where(below_and_falling, -1.0, 1.0)
    h = (direction * step)[:, None]                   # signed step, one per ball

    points = np.full((n, 3), np.nan)
    times = np.full(n, np.nan)
    active = np.ones(n, dtype=bool)
    t = 0.0

    while active.any() and t < max_time:
        a = acceleration(velocity, drag)
        next_position = position + velocity * h + 0.5 * a * h**2
        next_velocity = velocity + a * h

        z, next_z = position[:, 2], next_position[:, 2]
        coming_down = (direction > 0) & (z > contact_height) & (next_z <= contact_height)
        going_back_up = (direction < 0) & (z <= contact_height) & (next_z > contact_height)
        crossed = active & (coming_down | going_back_up)

        if crossed.any():
            # interpolate inside this small step to find the exact crossing
            fraction = (z[crossed] - contact_height) / (z[crossed] - next_z[crossed])
            points[crossed] = (position[crossed]
                               + fraction[:, None] * (next_position[crossed] - position[crossed]))
            times[crossed] = direction[crossed] * (t + fraction * step)
            active &= ~crossed

        position, velocity = next_position, next_velocity
        t += step

        # Flying backward drag speeds the ball up instead of slowing it so a
        # far off sample that never reaches contact height runs away. '
        #Give up on any ball faster than MAX_SPEED
        active &= np.linalg.norm(velocity, axis=1) < MAX_SPEED
        velocity[~active] = 0.0      # finished balls stop moving so nothing overflows

    points[:, 2] = floor_height
    return points, times


def predict_landing(kalman_filter, floor_height=0.0, ball_radius=BALL_RADIUS,
                    drag=BALL_DRAG, n_samples=500, rng=None):
    """Landing point and its uncertainty from the filter's current state.

    Contact is when the center reaches floor_height + ball_radius.
    drag: BALL_DRAG for our ball or 0 for the old gravity only prediction

    Returns dict with
        'point': (3,) best guess landing spot on the floor
        'time': seconds from now until landing
        'covariance': (2, 2) uncertainty of the landing (x, y), in m²
        'samples': (n, 3) the sampled landing points (for plotting the spread)
    or None if the ball can't land from this state.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Best guess the filters current state, flown to the floor
    x, P = kalman_filter.get_state()  # copies: the live filter is not changed
    result = find_landing_point(x, floor_height, ball_radius, drag)
    if result is None:
        return None
    point, time_to_landing = result

    # Carry the uncertainty forward to landing time. P grows on the way
    # covering pushes (spin, a wrong drag number) the model doesn't know about
    future = propagate_trajectory(x, P, kalman_filter.sigma_a, time_to_landing, drag=drag)
    x_land = future['state'][-1]
    P_land = future['covariance'][-1]

    # Draw random states that are consistent with that uncertainty
    samples = rng.multivariate_normal(x_land, P_land, size=n_samples)

    # Land each one, their spread is the landing uncertainty
    if drag == 0:
        landings = []
        for sample in samples:
            sample_result = find_landing_point(sample, floor_height, ball_radius)
            if sample_result is not None:
                landings.append(sample_result[0])
        landings = np.array(landings)
    else:
        landings, times = fly_to_floor(samples, floor_height, ball_radius, drag)
        landings = landings[~np.isnan(times)]

    return {
        'point': point,
        'time': time_to_landing,
        'covariance': np.cov(landings[:, :2].T), # spread in x and y only bc z is the floor
        'samples': landings,
    }
