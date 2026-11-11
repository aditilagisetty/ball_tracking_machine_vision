"""Run the Kalman filter on simulated noisy throws and check it's honest (NEES).

Measurement noise comes from the ZED 2i stereo geometry (stereo_noise.py):
a few mm sideways but several cm in depth, growing with distance. The camera
numbers there are placeholders until we measure the real ones.
"""
# Some ai assistance used
import numpy as np

from ball_tracking_machine_vision.kalman_filter import SIGMA_A, BallKalmanFilter
from ball_tracking_machine_vision.simulation import (
    BALL_DRAG,
    FRAME_DT,
    add_stereo_noise,
    drop_frames,
    random_throw_conditions,
    simulate_throw,
)
from ball_tracking_machine_vision.stereo_noise import (
    depth_of,
    make_camera,
    measurement_covariance,
    stereo_noise_std,
)

# The simulated ZED 2i, built from the placeholder values in stereo_noise.py
# (focal length, baseline, pixel noise, where the camera is mounted).
# Shared by every function below so they all see the same camera.
CAMERA = make_camera()

# Fraction of frames the detector misses, and how many frames to skip at the
# start of each throw while the filter is still working out the velocity.
DROP_RATE = 0.1
SKIP_FRAMES = 5


# how honest is the filter about its own uncertainty

# Normalized Estimation Error Squared
# np.linalg.solve(P, error) computes P⁻¹ · error, the same as using the inverse but more numerically stable.
# The result is eᵀ P⁻¹ e: each part of the error divided by how uncertain the filter said it was in that direction
# squared and added up

# For a 6-number state, an honest filter averages about 6:
# Bigger than 6: the errors are larger than the filter claimed --> overconfident.
# Smaller than 6: the filter claimed more uncertainty than it actually had --> too cautious.

# P is the filters covariance matrix
# 6 number state because we have position (x, y, z) and velocity (vx, vy, vz)
def nees(error, P):
    """Normalized estimation error squared. This tells us how big the error is
    measured in units of the filter's own uncertainty."""
    return error @ np.linalg.solve(P, error)


def simulate_measurements(seed, drag):
    """One simulated throw and what the camera would have measured of it.

    Returns (throw, meas):
        throw: the truth (positions, velocities, landing point)
        meas:  noisy positions, their timestamps, and the R matrix for each frame
    """
    # Each seed gives a different but repeatable throw.
    # Seed 7 is always the same throw so results can be compared fairly when sigma_a changes
    rng = np.random.default_rng(seed)
    throw = simulate_throw(*random_throw_conditions(rng), dt=FRAME_DT, drag=drag) # unpacks the (position, velocity) into 2 arguments

    # how much noise the simulated camera actually adds
    # stereo noise small sideways and large in depth, then some frames missed
    meas = drop_frames(add_stereo_noise(throw, CAMERA, rng), DROP_RATE, rng)
    return throw, meas


def run_filter(meas, sigma_a):
    """Run the filter over a throw's measurements, yielding (t, x, P) after each frame."""
    # amount of noise the filter is told the camera has:
    # meas['covariance'] holds one R per frame, computed with the same camera
    # model that made the noise, so here what the filter is told matches what
    # the camera actually does. The R given to __init__ is only a fallback
    # for calls that don't pass their own R; every call below does.
    kf = BallKalmanFilter(sigma_a, meas['covariance'][0])

    # R is worked out at the filters predicted position not the measured one
    def noise(position):
        return measurement_covariance(position, CAMERA)

    kf.initialize(meas['position'][0], meas['t'][0], R=noise)
    for t, z in zip(meas['t'][1:], meas['position'][1:]):
        kf.step(z, t, R=noise)
        x, P = kf.get_state()
        yield t, x, P   # hand back this frame's estimate, then carry on with the next one


def throw_nees(seed, sigma_a, drag):
    """Filter one simulated throw and return the NEES at every frame after the first few."""
    throw, meas = simulate_measurements(seed, drag)

    values = []
    for k, (t, x, P) in enumerate(run_filter(meas, sigma_a)): # frame counter k along (t, x, P)
        if k < SKIP_FRAMES: # skips first 5 frames -- filter is working out vel, including it will distort average
            continue
        i = np.argmin(np.abs(throw['t'] - t)) # finds true frame matching this measurement
        true_state = np.concatenate([throw['position'][i], throw['velocity'][i]])
        values.append(nees(true_state - x, P)) # real error only measurable in simulation
    return values


def average_nees(sigma_a, drag, n_throws=50):
    """Average NEES over many simulated throws."""
    values = []
    for seed in range(n_throws):
        values.extend(throw_nees(seed, sigma_a, drag))
    return np.mean(values)


def single_throw_demo(sigma_a=SIGMA_A):
    """Filter one throw and compare the final estimate to the truth."""
    throw, meas = simulate_measurements(seed=0, drag=BALL_DRAG)

    # How noisy the simulated camera is where this throw starts and ends.
    # stereo_noise_std returns (sideways, sideways, depth) standard deviations.
    for label, position in [('start', throw['position'][0]), ('landing', throw['landing_point'])]:
        depth = depth_of(position, CAMERA)
        sigma = stereo_noise_std(depth, CAMERA)
        print(f'{label:<8} {depth:.2f} m from camera: '
              f'sideways noise {sigma[0] * 1000:.1f} mm, depth noise {sigma[2] * 1000:.1f} mm')

    # Run through every frame; t, x, P end up holding the last frame's values.
    for t, x, P in run_filter(meas, sigma_a):
        pass

    i = np.argmin(np.abs(throw['t'] - t)) # true velocity at the same moment as the last estimate
    print('estimated velocity:', x[3:])
    print('true velocity:     ', throw['velocity'][i])
    print('velocity σ:        ', np.sqrt(np.diag(P)[3:]))


def main():
    single_throw_demo()
    print()

    # drag is REALITY (how the simulated ball flies); the filter never knows about it.
    # sigma_a is BELIEF (how wrong the filter thinks its gravity-only model might be).
    print('drag    sigma_a   NEES (ideal 6)')
    for drag in [0.0, BALL_DRAG]:
        for sigma_a in [1.5, SIGMA_A, 3.5]:
            print(f'{drag:<8.3f}{sigma_a:<10}{average_nees(sigma_a, drag):.2f}')


if __name__ == '__main__':
    main()