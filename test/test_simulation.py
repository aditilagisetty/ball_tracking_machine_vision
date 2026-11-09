"""Checks that the simulated throws obey the physics we expect."""

import numpy as np

from ball_tracking_machine_vision.simulation import (
    GRAVITY,
    BALL_DRAG,
    BALL_RADIUS,
    add_measurement_noise,
    add_stereo_noise,
    drop_frames,
    simulate_throw,
)
from ball_tracking_machine_vision.stereo_noise import (
    make_camera,
    measurement_covariance,
    stereo_noise_std,
)


def test_landing_matches_closed_form():
    """Without drag, landing time and point should match the projectile formula."""
    p0 = np.array([0.0, 0.0, 1.5])
    v0 = np.array([3.0, 0.5, 2.0])
    throw = simulate_throw(p0, v0, dt=1 / 60)

    # The throw ends when the center is one ball radius above the floor.
    drop = p0[2] - BALL_RADIUS
    t_land = (v0[2] + np.sqrt(v0[2]**2 + 2 * GRAVITY * drop)) / GRAVITY
    assert np.isclose(throw['landing_time'], t_land, atol=1e-6)
    assert np.allclose(throw['landing_point'][:2], p0[:2] + v0[:2] * t_land, atol=1e-6)


def test_frames_stay_above_floor():
    throw = simulate_throw([0, 0, 1.0], [2, 0, 3], dt=1 / 60)
    assert np.all(throw['position'][:, 2] > 0)
    assert np.allclose(np.diff(throw['t']), 1 / 60)


def test_drag_shortens_throw():
    p0, v0 = [0, 0, 1.2], [8, 0, 4]
    no_drag = simulate_throw(p0, v0, dt=1 / 60)
    with_drag = simulate_throw(p0, v0, dt=1 / 60, drag=BALL_DRAG)
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


def test_stereo_noise_is_worst_in_depth():
    """A camera looking along world +y should put its big depth noise on world y."""
    camera = make_camera(position=[0, -2, 1], target=[0, 0, 1])
    R = measurement_covariance(np.array([0.0, 0.0, 1.0]), camera)
    sigma = np.sqrt(np.diag(R))
    assert sigma[1] > 10 * sigma[0]      # depth (world y) much noisier than sideways (world x)
    assert np.isclose(sigma[0], sigma[2])  # the two sideways axes match


def test_stereo_noise_grows_with_distance():
    """Depth noise should grow with distance squared, sideways noise linearly."""
    camera = make_camera()
    near = stereo_noise_std(1.0, camera)
    far = stereo_noise_std(2.0, camera)
    assert np.isclose(far[0] / near[0], 2.0)
    assert np.isclose(far[2] / near[2], 4.0)


def test_add_stereo_noise_matches_covariance():
    """The noise actually added should have the spread the R matrices claim."""
    rng = np.random.default_rng(0)
    camera = make_camera(position=[0, -2, 1], target=[0, 0, 1])
    throw = simulate_throw([0, 0, 1.0], [0, 0, 0], dt=1 / 1000)   # ball barely moves
    noisy = add_stereo_noise(throw, camera, rng)
    spread = np.std(noisy['position'] - throw['position'], axis=0)
    # R at the true position the saved R is worked out at the noisy
    # measured position which is off when the depth noise is large
    expected = np.sqrt(np.diag(measurement_covariance(throw['position'][0], camera)))
    assert np.allclose(spread, expected, rtol=0.15)
