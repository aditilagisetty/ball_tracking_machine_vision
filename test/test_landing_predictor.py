import numpy as np

from ball_tracking_machine_vision.landing_predictor import (
    find_landing_point,
    fly_to_floor,
    propagate_trajectory,
)
from ball_tracking_machine_vision.simulation import BALL_DRAG, simulate_throw

START = np.array([0.0, 0.0, 1.3, 3.0, 0.2, 3.0])


def test_drag_landing_matches_simulation():
    """with drag on, the predicted landing should match the simulated ball"""
    throw = simulate_throw(START[:3], START[3:], dt=1 / 60, drag=BALL_DRAG)
    point, t = find_landing_point(START, drag=BALL_DRAG)
    assert np.linalg.norm(point - throw['landing_point']) < 0.005    # within 5 mm
    assert abs(t - throw['landing_time']) < 0.005


def test_zero_drag_flight_matches_formula():
    """flying with no drag should give the same answer as the exact formula"""
    points, times = fly_to_floor(START[None], drag=0.0)
    point, t = find_landing_point(START)
    assert np.linalg.norm(points[0] - point) < 0.001
    assert abs(times[0] - t) < 0.001


def test_ball_already_below_contact_flies_backward():
    """a ball just past contact height should land a little in the past"""
    _, t_land = find_landing_point(START, drag=BALL_DRAG)
    later = propagate_trajectory(START, np.zeros((6, 6)), 0.0, t_land + 0.02,
                                 drag=BALL_DRAG)['state'][-1]
    point, t = find_landing_point(later, drag=BALL_DRAG)
    assert abs(t - (-0.02)) < 0.002
