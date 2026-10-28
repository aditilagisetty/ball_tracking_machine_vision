"""Checks that the simulated throws obey the physics we expect."""

import numpy as np

from ball_tracking_machine_vision.simulation import (
    GRAVITY,
    TENNIS_BALL_DRAG,
    add_measurement_noise,
    drop_frames,
    simulate_throw,
)


def test_landing_matches_closed_form():
    """Without drag, landing time and point should match the projectile formula."""
    p0 = np.array([0.0, 0.0, 1.5])
    v0 = np.array([3.0, 0.5, 2.0])
    throw = simulate_throw(p0, v0, dt=1 / 60)

    t_land = (v0[2] + np.sqrt(v0[2]**2 + 2 * GRAVITY * p0[2])) / GRAVITY
    assert np.isclose(throw['landing_time'], t_land, atol=1e-6)
    assert np.allclose(throw['landing_point'][:2], p0[:2] + v0[:2] * t_land, atol=1e-6)


def test_frames_stay_above_floor():
    throw = simulate_throw([0, 0, 1.0], [2, 0, 3], dt=1 / 60)
    assert np.all(throw['position'][:, 2] > 0)
    assert np.allclose(np.diff(throw['t']), 1 / 60)


def test_drag_shortens_throw():
    p0, v0 = [0, 0, 1.2], [8, 0, 4]
    no_drag = simulate_throw(p0, v0, dt=1 / 60)
    with_drag = simulate_throw(p0, v0, dt=1 / 60, drag=TENNIS_BALL_DRAG)
    assert with_drag['landing_point'][0] < no_drag['landing_point'][0]


def test_noise_has_requested_spread():
    rng = np.random.default_rng(0)
    throw = simulate_throw([0, 0, 1.5], [0, 0, 0], dt=1 / 1000)
    noisy = add_measurement_noise(throw, noise_std=[0.01, 0.01, 0.03], rng=rng)
    spread = np.std(noisy['position'] - throw['position'], axis=0)
    assert np.allclose(spread, [0.01, 0.01, 0.03], rtol=0.15)


def test_drop_frames_keeps_first():
    rng = np.random.default_rng(1)
    throw = simulate_throw([0, 0, 1.5], [3, 0, 2], dt=1 / 60)
    noisy = add_measurement_noise(throw, 0.01, rng)
    dropped = drop_frames(noisy, drop_rate=0.5, rng=rng)
    assert dropped['t'][0] == 0.0
    assert len(dropped['t']) < len(noisy['t'])
