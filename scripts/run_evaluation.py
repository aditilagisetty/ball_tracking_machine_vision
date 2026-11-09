"""run the evaluation and print the landing error table
    PYTHONPATH=. .venv/bin/python scripts/run_evaluation.py
"""

import numpy as np

from ball_tracking_machine_vision.evaluation import (
    error_vs_fraction_observed,
    evaluate_throw,
    print_table,
)
from ball_tracking_machine_vision.kalman_filter import SIGMA_A
from ball_tracking_machine_vision.simulation import (
    BALL_DRAG,
    add_stereo_noise,
    drop_frames,
    random_throw_conditions,
    simulate_throw,
)
from ball_tracking_machine_vision.stereo_noise import make_camera

N_THROWS = 20
DROP_RATE = 0.1


def simulated_throw(seed, camera):
    """one fake throw: (measurements, real landing spot, landing time)"""
    rng = np.random.default_rng(seed)
    throw = simulate_throw(*random_throw_conditions(rng), dt=1 / 60, drag=BALL_DRAG)
    meas = drop_frames(add_stereo_noise(throw, camera, rng), DROP_RATE, rng)
    return meas, throw['landing_point'], throw['landing_time']


def main():
    camera = make_camera()
    all_rows = []
    for seed in range(N_THROWS):
        meas, true_landing, landing_time = simulated_throw(seed, camera)
        rng = np.random.default_rng(1000 + seed)   # for the random landing samples
        all_rows.append(evaluate_throw(meas, true_landing, landing_time,
                                       sigma_a=SIGMA_A, rng=rng))

    print(f'{N_THROWS} simulated throws, median landing error\n')
    print_table(error_vs_fraction_observed(all_rows))


if __name__ == '__main__':
    main()
