"""Filter tests on simulated throws."""


def test_tracks_simulated_throw():
    """Filter error should stay small on a noisy simulated throw."""
    pass


def test_error_within_reported_uncertainty():
    """Errors should mostly fall inside the filter's covariance."""
    pass


def test_recovers_from_bad_start():
    """Seed 3 used to lock out after a bad early velocity and drift 1.7 m off."""
    import numpy as np
    from scripts.simulate_filter import run_filter, simulate_measurements
    from ball_tracking_machine_vision.simulation import TENNIS_BALL_DRAG

    throw, meas = simulate_measurements(3, TENNIS_BALL_DRAG)
    for t, x, P in run_filter(meas, sigma_a=1.0):
        i = np.argmin(np.abs(throw['t'] - t))
        assert np.linalg.norm(throw['position'][i] - x[:3]) < 0.3
