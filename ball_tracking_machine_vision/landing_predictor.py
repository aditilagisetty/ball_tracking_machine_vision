"""Extrapolating the filter state forward to find where the ball lands.

The filter tracks the ball's CENTER, so the ball "lands" when its center is
one radius above the floor (floor_height + ball_radius), the same rule
simulate_throw uses. The landing point reported is the spot on the floor
directly below the center at that moment (z = floor_height), which is what
the 10 cm floor grid measures.
"""

from ball_tracking_machine_vision.simulation import BALL_RADIUS


def propagate_trajectory(state, covariance, dt, max_time):
    """Step the state forward and return future positions with uncertainty."""
    pass


def find_landing_point(trajectory, floor_height=0.0, ball_radius=BALL_RADIUS):
    """Return where and when the trajectory touches the floor.

    Contact is when the center reaches floor_height + ball_radius.
    """
    pass


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
