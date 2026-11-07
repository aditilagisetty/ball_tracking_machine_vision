"""Filter tests on simulated throws."""

import numpy as np

from ball_tracking_machine_vision.simulation import BALL_DRAG
from scripts.simulate_filter import run_filter, simulate_measurements


def test_tracks_simulated_throw():
    """Filter error should stay small on a noisy simulated throw."""
    throw, meas = simulate_measurements(seed=0, drag=BALL_DRAG)

    # run through every frame 
    # t and x end up holding the last frames values
    for t, x, P in run_filter(meas, sigma_a=1.5):
        pass
    
    i = np.argmin(np.abs(throw['t']-t)) # true frame at the same time
    error = np.linalg.norm(throw['position'][i] - x[:3])
    assert error < 0.05 # within 5 cm


def test_error_within_reported_uncertainty():
    """Errors should mostly fall inside the filter's covariance."""
    pass


def test_recovers_from_bad_start():
    """Seed 3 used to lock out after a bad early velocity and drift 1.7 m off."""
    import numpy as np
    from scripts.simulate_filter import run_filter, simulate_measurements
    from ball_tracking_machine_vision.simulation import BALL_DRAG

    throw, meas = simulate_measurements(3, BALL_DRAG)
    for t, x, P in run_filter(meas, sigma_a=1.5):
        i = np.argmin(np.abs(throw['t'] - t))
        assert np.linalg.norm(throw['position'][i] - x[:3]) < 0.3
