"""run the pipeline on one recorded throw, print what happened, and save the overlay and plots

run from the repo folder, with a video file or a throw folder from scripts/record.py:
    PYTHONPATH=. .venv/bin/python scripts/run_throw.py recordings/goodvideo3.mp4 --start 30 --end 40
    PYTHONPATH=. .venv/bin/python scripts/run_throw.py data/throw_01

results go to output/<throw name>/: overlay.mp4, track_3d.png, landing_uncertainty.png
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')   # save plots to files no window needed
import matplotlib.pyplot as plt  
import numpy as np  

from ball_tracking_machine_vision.camera_capture import rectify
from ball_tracking_machine_vision.landing_predictor import propagate_trajectory
from ball_tracking_machine_vision.pipeline import (
    calibration_for,
    throw_frames,
    track_throw,
    video_fps,
)
from ball_tracking_machine_vision.simulation import BALL_DRAG
from ball_tracking_machine_vision.visualization import (
    draw_detection,
    draw_predicted_path,
    plot_landing_uncertainty,
    plot_track_3d,
)


def print_track(track):
    """one line per frame: was the ball found, where, and where we think it lands"""
    print('frame  ball found   world position (m)       NIS   used   predicted landing (m)')
    for i, frame in enumerate(track['frame']):
        found = 'yes' if track['detection'][i] is not None else 'no'
        p = track['position'][i]
        where = '' if p is None else f'({p[0]:5.2f}, {p[1]:5.2f}, {p[2]:5.2f})'
        nis = '' if track['nis'][i] is None else f"{track['nis'][i]:5.1f}"
        used = {True: 'yes', False: 'no', None: ''}[track['accepted'][i]]
        pred = track['prediction'][i]
        land = '' if pred is None else f"({pred['point'][0]:.2f}, {pred['point'][1]:.2f})"
        print(f'{frame:5d}  {found:<10}  {where:<23}  {nis:>5}  {used:>4}   {land}')

    n_measured = len(track['measurements']['t'])
    print(f"\nball found in both images in {sum(d is not None for d in track['detection'])}"
          f" of {len(track['frame'])} frames, 3D position in {n_measured}")

    # only measurements the filter used; rejected ones are wrong detections
    nis = [v for v, used in zip(track['nis'], track['accepted']) if v is not None and used]
    rejected = sum(used is False for used in track['accepted'])
    print(f'{rejected} measurements thrown out as too far from the prediction')
    if nis:
        # about 3 to 4 if the noise settings are right. much bigger means the
        # filter trusts the camera or its physics too much: raise PIXEL_NOISE or SIGMA_A
        print(f'average NIS {np.mean(nis):.1f} (want about 3 to 4)')


def save_overlay(video, start, end, track, calibration, path, fps):
    """overlay video: detection and predicted path drawn on the rectified left image"""
    camera = calibration['camera']
    camera_pose = (camera['rotation'], camera['position'])
    writer = None
    for i, (frame_number, left, right, t) in enumerate(throw_frames(video, start, end)):
        image, _ = rectify(left, right, calibration['maps'])

        detection = track['detection'][i]
        image = draw_detection(image, None if detection is None else detection[0])

        prediction = track['prediction'][i]
        if prediction is not None:
            # the predicted flight to the floor with drag like the landing prediction
            path_ahead = propagate_trajectory(track['estimate'][i], np.zeros((6, 6)), 0.0,
                                              prediction['time'], drag=BALL_DRAG)['state']
            image = draw_predicted_path(image, path_ahead[:, :3], calibration['P1'], camera_pose)

        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (width, height))
        writer.write(image)

    if writer is not None:
        writer.release()
        print(f'saved {path}')


def save_plots(track, out_dir):
    """3d track and landing uncertainty from the last prediction"""
    estimates = [e for e in track['estimate'] if e is not None]
    predictions = [p for p in track['prediction'] if p is not None]
    if not estimates or not predictions:
        print('(no plots: the filter never got a prediction)')
        return

    fig, _ = plot_track_3d(track['measurements']['position'], np.array(estimates), predictions[-1])
    fig.savefig(out_dir / 'track_3d.png', dpi=200)
    plt.close(fig)
    print(f"saved {out_dir / 'track_3d.png'}")

    last = predictions[-1]
    fig, _ = plot_landing_uncertainty(last['point'], last['covariance'])
    fig.savefig(out_dir / 'landing_uncertainty.png', dpi=200)
    plt.close(fig)
    print(f"saved {out_dir / 'landing_uncertainty.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='a video file or a throw folder from scripts/record.py')
    parser.add_argument('--start', type=int, default=0, help='first frame in the air')
    parser.add_argument('--end', type=int, default=None, help='last frame in the air')
    parser.add_argument('--method', default='color', choices=['color', 'hough'])
    args = parser.parse_args()

    calibration = calibration_for(args.video)   # uses camera_pose.json next to the video if there is one
    track = track_throw(throw_frames(args.video, args.start, args.end), calibration,
                        method=args.method, rng=np.random.default_rng(0))
    print_track(track)

    out_dir = Path('output') / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = video_fps(args.video)
    save_overlay(args.video, args.start, args.end, track, calibration,
                 out_dir / 'overlay.mp4', fps)
    save_plots(track, out_dir)


if __name__ == '__main__':
    main()
