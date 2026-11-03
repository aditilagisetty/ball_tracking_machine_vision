"""Run the Kalman filter on simulated noisy throws and plot the results."""

import numpy as np

from ball_tracking_machine_vision.kalman_filter import BallKalmanFilter
from ball_tracking_machine_vision.simulation import (
    add_measurement_noise,
    drop_frames,
    simulate_throw,
)


def main():
    rng = np.random.default_rng(0)
    noise_std = np.array([0.01, 0.01, 0.03])

    throw = simulate_throw([0, 0, 1.5], [3, 0.3, 2.5], dt=1 / 60)
    meas = drop_frames(add_measurement_noise(throw, noise_std, rng), 0.1, rng)

    kf = BallKalmanFilter(sigma_a=2.0, measurement_noise=np.diag(noise_std**2))
    kf.initialize(meas['position'][0], meas['t'][0])
    for t, z in zip(meas['t'][1:], meas['position'][1:]):
        kf.step(z, t)

    x, P = kf.get_state()
    print('estimated velocity:', x[3:])
    print('true velocity:     ', throw['velocity'][-1])
    print('velocity σ:        ', np.sqrt(np.diag(P)[3:]))


if __name__ == '__main__':
    main()