"""run the evaluation and print the landing error table


the error plot is saved to output/ once visualization.plot_error_vs_fraction is written
"""

import argparse
from pathlib import Path

import numpy as np

from ball_tracking_machine_vision.evaluation import (
    error_vs_fraction_observed,
    evaluate_all,
    evaluate_throw,
    print_table,
)
from ball_tracking_machine_vision.kalman_filter import SIGMA_A
from ball_tracking_machine_vision.simulation import (
    BALL_DRAG,
    FRAME_DT,
    add_stereo_noise,
    drop_frames,
    random_throw_conditions,
    simulate_throw,
)
from ball_tracking_machine_vision.stereo_noise import make_camera
from ball_tracking_machine_vision.visualization import plot_error_vs_fraction

# 200 so the medians are steady with 20 the early fractions jump around a lot
N_THROWS = 200
OUTPUT_DIR = Path('output')
DROP_RATE = 0.1


def simulated_throw(seed, camera):
    """one fake throw: (measurements, real landing spot, landing time)"""
    rng = np.random.default_rng(seed)
    throw = simulate_throw(*random_throw_conditions(rng), dt=FRAME_DT, drag=BALL_DRAG)
    meas = drop_frames(add_stereo_noise(throw, camera, rng), DROP_RATE, rng)
    return meas, throw['landing_point'], throw['landing_time']


def simulated_table():
    """error table over N_THROWS simulated throws"""
    camera = make_camera()
    all_rows = []
    for seed in range(N_THROWS):
        meas, true_landing, landing_time = simulated_throw(seed, camera)
        rng = np.random.default_rng(1000 + seed)   # for the random landing samples
        all_rows.append(evaluate_throw(meas, true_landing, landing_time,
                                       sigma_a=SIGMA_A, rng=rng, camera=camera))
    return error_vs_fraction_observed(all_rows)


def save_plot(table, name):
    """save the error plot if visualization.plot_error_vs_fraction is written yet"""
    fig = plot_error_vs_fraction(table)
    if fig is None:
        print('(no plot yet: visualization.plot_error_vs_fraction is still a stub)')
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200)
    print(f'saved {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', help='use the recorded throws')
    parser.add_argument('--videos', default='recordings', help='folder with the videos')
    parser.add_argument('--truth', default='recordings/ground_truth.csv')
    args = parser.parse_args()

    if args.real:
        table, rows = evaluate_all(args.videos, args.truth, sigma_a=SIGMA_A)
        print(f'{len(rows)} recorded throws, median landing error\n')
        print_table(table)
        save_plot(table, 'error_vs_fraction_real.png')
    else:
        table = simulated_table()
        print(f'{N_THROWS} simulated throws, median landing error\n')
        print_table(table)
        save_plot(table, 'error_vs_fraction_sim.png')


if __name__ == '__main__':
    main()
