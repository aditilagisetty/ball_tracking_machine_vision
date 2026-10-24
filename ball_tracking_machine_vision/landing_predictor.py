"""Extrapolating the filter state forward to find where the ball lands."""


def propagate_trajectory(state, covariance, dt, max_time):
    """Step the state forward and return future positions with uncertainty."""
    pass


def find_landing_point(trajectory, floor_height):
    """Return where and when the trajectory crosses the floor."""
    pass


def predict_landing(kalman_filter, floor_height):
    """Landing point and its uncertainty from the filter's current state."""
    pass
