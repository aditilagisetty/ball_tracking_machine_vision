import numpy as np

from ball_tracking_machine_vision.evaluation import (
    error_vs_fraction_observed,
    evaluate_throw,
    landing_error,
)
from ball_tracking_machine_vision.simulation import (
    BALL_DRAG,
    FRAME_DT,
    add_stereo_noise,
    simulate_throw,
)
from ball_tracking_machine_vision.stereo_noise import make_camera


def test_landing_error_ignores_height():
    assert landing_error([3.0, 4.0, 0.0], [0.0, 0.0, 0.0]) == 5.0
    assert landing_error([1.0, 1.0, 0.5], [1.0, 1.0, 0.0]) == 0.0


def test_error_shrinks_as_more_flight_is_seen():
    """watching more of the throw should make both guesses better"""
    camera = make_camera()
    rows = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        throw = simulate_throw([0, 0, 1.1], [1.8, 0.2, 2.5], dt=FRAME_DT, drag=BALL_DRAG)
        meas = add_stereo_noise(throw, camera, rng)
        rows.append(evaluate_throw(meas, throw['landing_point'], throw['landing_time'],
                                   fractions=[0.25, 0.9], rng=rng, camera=camera))
    table = error_vs_fraction_observed(rows, fractions=[0.25, 0.9])

    early, late = table
    assert late['filter_median'] < early['filter_median']
    assert late['baseline_median'] < early['baseline_median']
    # at VGA with 2.5 px of detector noise one measurement is about 18 cm off
    # in depth at 1.5 m so this is loose within 25 cm after 90% of the flight
    assert late['filter_median'] < 0.25
