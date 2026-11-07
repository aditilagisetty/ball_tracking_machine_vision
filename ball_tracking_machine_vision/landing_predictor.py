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


def predict_landing(kalman_filter, floor_height=0.0, ball_radius=BALL_RADIUS):
    """Landing point and its uncertainty from the filter's current state.

    Contact is when the center reaches floor_height + ball_radius.
    """
    pass
