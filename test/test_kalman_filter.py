"""Filter tests on simulated throws."""

import numpy as np

from ball_tracking_machine_vision.kalman_filter import SIGMA_A
from ball_tracking_machine_vision.simulation import BALL_DRAG
from scripts.simulate_filter import run_filter, simulate_measurements


def test_tracks_simulated_throw():
    """Filter error should stay small on a noisy simulated throw."""
    throw, meas = simulate_measurements(seed=0, drag=BALL_DRAG)

    # run through every frame 
    # t and x end up holding the last frames values
    for t, x, P in run_filter(meas, sigma_a=SIGMA_A):
        pass
    
    i = np.argmin(np.abs(throw['t']-t)) # true frame at the same time
    error = throw['position'][i] - x[:3]

    # at VGA the cameras depth noise is about 12 cm at 2 m so a fixed 5 cm
    # limit is too strict the error should be inside the filters
    # own 99% uncertainty (chi-square, 3 numbers) and never more than 20 cm
    assert error @ np.linalg.solve(P[:3, :3], error) < 11.34
    assert np.linalg.norm(error) < 0.20


def test_error_within_reported_uncertainty():
    """Errors should mostly fall inside the filter's covariance."""
    inside = []
    for seed in range(30):
        throw, meas = simulate_measurements(seed, BALL_DRAG)
        for k, (t, x, P) in enumerate(run_filter(meas, sigma_a=SIGMA_A)):
            if k < 5: # skip startup frames
                continue
            i = np.argmin(np.abs(throw['t'] - t))
            error = np.abs(throw['position'][i] - x[:3])
            sigma = np.sqrt(np.diag(P)[:3]) # filters claimed sigma for x, y, z
            inside.append(error < 2 * sigma)

    fraction = np.mean(inside) # ideal is about 0.95
    assert 0.90 < fraction < 0.99


def test_recovers_from_bad_start():
    """Seed 3 used to lock out after a bad early velocity and drift 1.7 m off."""
    import numpy as np
    from scripts.simulate_filter import run_filter, simulate_measurements
    from ball_tracking_machine_vision.simulation import BALL_DRAG

    throw, meas = simulate_measurements(3, BALL_DRAG)
    for t, x, P in run_filter(meas, sigma_a=SIGMA_A):
        i = np.argmin(np.abs(throw['t'] - t))
        assert np.linalg.norm(throw['position'][i] - x[:3]) < 0.3
