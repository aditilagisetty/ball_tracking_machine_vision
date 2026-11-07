"""Simulated throws with known physics, for testing the filter.

World frame: z up, origin on the floor, meters and seconds. This is the same
frame the perception side hands to the filter, so simulated and real throws
can go through the same code.
"""

import numpy as np

from ball_tracking_machine_vision.stereo_noise import (
    depth_of,
    measurement_covariance,
    stereo_noise_std,
)

GRAVITY = 9.81

# Quadratic drag coefficient for a tennis ball, k = 0.5 * rho * Cd * A / m
# (air density 1.2 kg/m^3, Cd ~0.55, radius 3.35 cm, mass 58 g). The drag
# acceleration is -k * |v| * v, so about 2 m/s^2 at 10 m/s.
TENNIS_BALL_DRAG = 0.02


def simulate_throw(initial_position, initial_velocity, dt, floor_height=0.0,
                   drag=0.0, substeps=10, max_time=10.0):
    """Generate a true ballistic trajectory until it hits the floor.

    Args:
        initial_position: starting (x, y, z), must be above the floor.
        initial_velocity: starting (vx, vy, vz).
        dt: time between camera frames (e.g. 1/60).
        floor_height: z of the floor.
        drag: quadratic drag coefficient (0 for gravity only,
            TENNIS_BALL_DRAG for a realistic ball).
        substeps: integration steps per frame. Only matters when drag > 0;
            with gravity only each step is exact.
        max_time: safety limit so a bad input can't loop forever.

    Returns:
        dict with
            't': (N,) frame timestamps
            'position': (N, 3) true positions at each frame
            'velocity': (N, 3) true velocities at each frame
            'landing_point': (3,) where the ball crosses the floor
            'landing_time': time it crosses the floor
    """
    position = np.asarray(initial_position, dtype=float)
    velocity = np.asarray(initial_velocity, dtype=float)
    if position[2] <= floor_height:
        raise ValueError('initial position must be above the floor')

    h = dt / substeps
    gravity = np.array([0.0, 0.0, -GRAVITY])

    times = [0.0]
    positions = [position.copy()]
    velocities = [velocity.copy()]
    t = 0.0

    while t < max_time:
        for _ in range(substeps):
            acceleration = gravity - drag * np.linalg.norm(velocity) * velocity
            next_position = position + velocity * h + 0.5 * acceleration * h**2
            next_velocity = velocity + acceleration * h

            if next_position[2] <= floor_height:
                # Interpolate within this small step to find the exact crossing.
                fraction = (position[2] - floor_height) / (position[2] - next_position[2])
                landing_point = position + fraction * (next_position - position)
                landing_point[2] = floor_height
                return {
                    't': np.array(times),
                    'position': np.array(positions),
                    'velocity': np.array(velocities),
                    'landing_point': landing_point,
                    'landing_time': t + fraction * h,
                }

            position, velocity = next_position, next_velocity
            t += h

        times.append(t)
        positions.append(position.copy())
        velocities.append(velocity.copy())

    raise RuntimeError('ball did not reach the floor within max_time')


def random_throw_conditions(rng, floor_height=0.0):
    """Random but realistic starting position and velocity for a hand toss.

    Released around chest height, thrown mostly along +x with some sideways
    spread and an upward arc. Returns (initial_position, initial_velocity).
    """
    initial_position = np.array([
        rng.uniform(-0.2, 0.2),
        rng.uniform(-0.2, 0.2),
        floor_height + rng.uniform(1.0, 1.6),
    ])
    initial_velocity = np.array([
        rng.uniform(2.0, 4.0),
        rng.uniform(-0.5, 0.5),
        rng.uniform(1.5, 4.0),
    ])
    return initial_position, initial_velocity


def add_measurement_noise(trajectory, noise_std, rng):
    """Add Gaussian noise to simulate camera measurements.

    Args:
        trajectory: output of simulate_throw.
        noise_std: one standard deviation for all axes, or (σx, σy, σz).
            Stereo is noisiest along the camera's depth axis, so give that
            axis the largest value.
        rng: numpy Generator, e.g. np.random.default_rng(0).

    Returns:
        dict with 't': (N,) timestamps and 'position': (N, 3) noisy positions.
    """
    true_positions = trajectory['position']
    noise = rng.normal(0.0, 1.0, size=true_positions.shape) * np.asarray(noise_std)
    return {
        't': trajectory['t'].copy(),
        'position': true_positions + noise,
    }


def add_stereo_noise(trajectory, camera, rng):
    """Add noise the way a stereo camera would: small sideways, large in depth.

    Unlike add_measurement_noise, the noise here depends on where the ball is:
    it's worse the further the ball is from the camera, and worst along the
    camera's viewing direction (see stereo_noise.py).

    Args:
        trajectory: output of simulate_throw.
        camera: output of stereo_noise.make_camera.
        rng: numpy Generator, e.g. np.random.default_rng(0).

    Returns:
        dict with
            't': (N,) timestamps
            'position': (N, 3) noisy positions
            'covariance': (N, 3, 3) the R matrix to give the filter for each
                frame. It's computed from the *measured* position, because
                the real pipeline won't know the true one either.
    """
    noisy_positions = []
    covariances = []
    for true_position in trajectory['position']:
        # Noise is generated in the camera frame, where it's easy to describe:
        # independent along right, down and forward, with forward much larger.
        sigma = stereo_noise_std(depth_of(true_position, camera), camera)
        noise_camera = rng.normal(0.0, 1.0, size=3) * sigma

        # Rotate the noise into the world frame and add it to the true position.
        noisy_position = true_position + camera['rotation'] @ noise_camera
        noisy_positions.append(noisy_position)

        covariances.append(measurement_covariance(noisy_position, camera))

    return {
        't': trajectory['t'].copy(),
        'position': np.array(noisy_positions),
        'covariance': np.array(covariances),
    }


def drop_frames(measurements, drop_rate, rng):
    """Randomly remove frames to mimic missed detections.

    The first frame is always kept so the filter has somewhere to start.
    The filter should handle the gaps by predicting without updating.
    Works on every per-frame array in the dict ('t', 'position', and
    'covariance' if present), so they all stay lined up.
    """
    keep = rng.random(len(measurements['t'])) >= drop_rate
    keep[0] = True
    return {key: values[keep] for key, values in measurements.items()}
