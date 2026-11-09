"""how good are the landing guesses

we stop watching the throw partway through, guess where it will land,
and check how far off the guess is. we do this at a few points in the
flight for both the kalman filter and the baseline

    't':          (N,) times
    'position':   (N, 3) measured ball centers in the world frame
    'covariance': (N, 3, 3) noise matrix for each measurement
"""

import numpy as np

from ball_tracking_machine_vision.baseline import (
    fit_trajectory,
    predict_landing_baseline,
)
from ball_tracking_machine_vision.kalman_filter import BallKalmanFilter
from ball_tracking_machine_vision.landing_predictor import predict_landing
from ball_tracking_machine_vision.simulation import BALL_RADIUS

# how much of the flight we watch before each guess
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 0.9]

# cutoff for inside the 95% ellipse with 2 numbers (landing x and y)
# a filter that is honest about its uncertainty should be inside about 95% of the time
COVERAGE_THRESHOLD_2D = 5.99


def load_ground_truth(path):
    """load the real landing spots"""
    # todo: depends on how we save the real throws
    pass


def landing_error(predicted, true_landing):
    """distance on the floor between the guess and the real landing (m)

    only x and y count since both points are on the floor
    """
    return np.linalg.norm(np.asarray(predicted)[:2] - np.asarray(true_landing)[:2])


def measurements_up_to(meas, cutoff_time):
    """only the measurements taken at or before cutoff_time"""
    keep = meas['t'] <= cutoff_time
    return {key: values[keep] for key, values in meas.items()}


def run_kalman_filter(meas, sigma_a):
    """run a new filter over these measurements and give it back"""
    kf = BallKalmanFilter(sigma_a, meas['covariance'][0])
    kf.initialize(meas['position'][0], meas['t'][0], R=meas['covariance'][0])
    for t, z, R in zip(meas['t'][1:], meas['position'][1:], meas['covariance'][1:]):
        kf.step(z, t, R=R)
    return kf


def evaluate_throw(meas, true_landing, landing_time, sigma_a=1.5,
                   fractions=FRACTIONS, floor_height=0.0,
                   ball_radius=BALL_RADIUS, rng=None):
    """filter and baseline errors for one throw at each fraction

    inputs
        meas: measurement dict (see top of file)
        true_landing: (3,) where the ball really landed
        landing_time: when it landed, same clock as meas['t']
        sigma_a: process noise for the filter
        fractions: how much of the flight to watch before each guess

    gives back a list with one dict per fraction
        'fraction'
        'filter_error', 'baseline_error': error in m, or None if no guess
        'filter_inside': True if the real landing was inside the filters 95% ellipse
    """
    rng = np.random.default_rng() if rng is None else rng
    t_start = meas['t'][0]
    rows = []

    for fraction in fractions:
        cutoff = t_start + fraction * (landing_time - t_start)
        seen = measurements_up_to(meas, cutoff)
        row = {'fraction': fraction, 'filter_error': None,
               'baseline_error': None, 'filter_inside': None}

        # kalman filter: track what we saw, then push it forward to the floor
        if len(seen['t']) >= 2:
            kf = run_kalman_filter(seen, sigma_a)
            prediction = predict_landing(kf, floor_height, ball_radius, rng=rng)
            if prediction is not None:
                row['filter_error'] = landing_error(prediction['point'], true_landing)

                # was the real landing inside the filters own 95% ellipse
                miss = np.asarray(true_landing)[:2] - prediction['point'][:2]
                d2 = miss @ np.linalg.solve(prediction['covariance'], miss)
                row['filter_inside'] = d2 < COVERAGE_THRESHOLD_2D

        # baseline: fit curves to the same measurements and extend them
        if len(seen['t']) >= 3:
            fit = fit_trajectory(seen['t'], seen['position'])
            result = predict_landing_baseline(fit, floor_height, ball_radius)
            if result is not None:
                row['baseline_error'] = landing_error(result[0], true_landing)

        rows.append(row)
    return rows


def error_vs_fraction_observed(throw_rows, fractions=FRACTIONS):
    """combine many throws into one table with the middle (median) error per fraction

    inputs
        throw_rows: one entry per throw, each one is what evaluate_throw gave back

    gives back a list with one dict per fraction
        'fraction'
        'filter_median', 'baseline_median': median error in m over throws that gave a guess
        'filter_failed', 'baseline_failed': how many throws gave no guess
        'filter_coverage': share of throws inside the filters 95% ellipse
        'n_throws'
    """
    table = []
    for i, fraction in enumerate(fractions):
        rows = [throw[i] for throw in throw_rows]
        filter_errors = [r['filter_error'] for r in rows if r['filter_error'] is not None]
        baseline_errors = [r['baseline_error'] for r in rows if r['baseline_error'] is not None]
        inside = [r['filter_inside'] for r in rows if r['filter_inside'] is not None]

        table.append({
            'fraction': fraction,
            'filter_median': np.median(filter_errors) if filter_errors else None,
            'baseline_median': np.median(baseline_errors) if baseline_errors else None,
            'filter_failed': len(rows) - len(filter_errors),
            'baseline_failed': len(rows) - len(baseline_errors),
            'filter_coverage': np.mean(inside) if inside else None,
            'n_throws': len(rows),
        })
    return table


def print_table(table):
    """print the error table in cm"""
    def cm(value):
        return '   --' if value is None else f'{value * 100:5.1f}'

    print('seen   filter (cm)  baseline (cm)  filter in 95%  baseline no answer')
    for row in table:
        coverage = '  --' if row['filter_coverage'] is None else f"{row['filter_coverage']:4.0%}"
        print(f"{row['fraction']:<6.0%} {cm(row['filter_median']):>11}  "
              f"{cm(row['baseline_median']):>13}  {coverage:>13}  "
              f"{row['baseline_failed']:>18}")


def evaluate_all(data_dir, ground_truth_path):
    """run the whole evaluation on all the real throws"""
    # todo once real throws exist: load each recording from data_dir,
    # match it to its real landing from load_ground_truth, run
    # evaluate_throw on each, then error_vs_fraction_observed
    pass
