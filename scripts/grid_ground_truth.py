"""camera pose and true landing spots from the checkerboard on the floor

run:
    PYTHONPATH=. .venv/bin/python scripts/grid_ground_truth.py
then check the printed table and run:
    PYTHONPATH=. .venv/bin/python scripts/run_evaluation.py --real --videos grid_recordings --truth grid_recordings/ground_truth.csv
"""
# Some ai assistance used

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from ball_tracking_machine_vision.camera_capture import rectify
from ball_tracking_machine_vision.pipeline import (
    calibration_for,
    find_ball_in_both,
    make_calibration,
    save_camera_pose,
    throw_frames,
)
from ball_tracking_machine_vision.simulation import BALL_RADIUS, GRAVITY
from ball_tracking_machine_vision.triangulation import camera_to_world, triangulate

SQUARE_SIZE = 3.1 * 0.0254        # m
BOARD_CORNERS = (9, 7)            # inner corners the detector finds
UPSCALE = 3                       # the board is small at VGA so look at it 3x bigger

# a stretch of track counts as free fall if height + g t^2 / 2 is a straight
# line in time to within this much (m). allows for stereo noise and some drag
FREE_FALL_TOLERANCE = 0.05


def video_number(path):
    """sort "video (10).mp4" after "video (9).mp4" """
    found = re.search(r'\((\d+)\)', path.name)
    return int(found.group(1)) if found else 0


def find_board(left_rectified):
    """the board's inner corners in the left image, or None"""
    gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    ok, corners = cv2.findChessboardCornersSB(
        big, BOARD_CORNERS, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 2) / UPSCALE if ok else None


def board_pose(videos, calibration):
    """up direction camera frame and camera height from every view of the board

    looks at the first few frames of each video, before the arm can block the board,
    and averages over every view where the whole board was found
    """
    K = calibration['P1'][:, :3]          # images are rectified so no distortion
    columns, rows = BOARD_CORNERS
    board = np.array([[i * SQUARE_SIZE, j * SQUARE_SIZE, 0.0]
                      for j in range(rows) for i in range(columns)])
    ups, heights = [], []
    for video in videos:
        for _, left, right, _ in throw_frames(video, 0, 5):
            left_rect, _ = rectify(left, right, calibration['maps'])
            corners = find_board(left_rect)
            if corners is None:
                continue
            ok, rvec, tvec = cv2.solvePnP(board, corners, K, None)
            rotation, _ = cv2.Rodrigues(rvec)          # board frame -> camera frame
            up = rotation[:, 2]                        # the board's normal
            if up[1] > 0:
                up = -up                               # camera y points down, so up has negative y
            ups.append(up)
            heights.append(abs(up @ tvec.ravel()))    # camera's distance to the board plane
    if not ups:
        raise RuntimeError('the board was not found in any video')
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)
    return up, float(np.mean(heights)), float(np.std(heights)), len(ups)


def track_ball(video, calibration):
    """per frame: time, left image detection (u, v, r) and world position, or None"""
    frames = {}
    for n, left, right, t in throw_frames(video):
        left_rect, right_rect = rectify(left, right, calibration['maps'])
        detection = find_ball_in_both(left_rect, right_rect, 'color')
        if detection is None:
            continue
        point = triangulate(detection[0], detection[1], calibration['P1'], calibration['P2'])
        if point is None:
            continue
        camera = calibration['camera']
        world = np.array(camera_to_world(point, (camera['rotation'], camera['position'])))
        frames[n] = {'t': t, 'uv': np.array(detection[0][:2]), 'world': world}
    return frames


def find_landing_frame(frames):
    """the ball's lowest point in the image, just before its first bounce

    the first frame that is low (center under 20 cm up), lower in the image than
    the tracked frame before it, and lower than the next one (on the way back up).
    a missing frame or two on either side is allowed: the ball is sometimes not
    found right before it lands
    """
    numbers = sorted(frames)
    for before, n, after in zip(numbers, numbers[1:], numbers[2:]):
        if n - before > 3 or after - n > 3:
            continue
        v_before, v, v_after = (frames[k]['uv'][1] for k in (before, n, after))
        if frames[n]['world'][2] < 0.2 and v > v_before and v >= v_after:
            return n
    return None


def find_release_frame(frames, landing):
    """earliest frame from which the height is free fall all the way to the landing"""
    release = None
    start = landing - 1
    while start - 1 in frames:
        start -= 1
        span = [k for k in range(start, landing) if k in frames]
        t = np.array([frames[k]['t'] for k in span])
        z = np.array([frames[k]['world'][2] for k in span])
        straight = z + 0.5 * GRAVITY * (t - t[0]) ** 2       # a line if only gravity acts
        fit = np.polyval(np.polyfit(t, straight, 1), t)
        if len(span) >= 3 and np.max(np.abs(straight - fit)) > FREE_FALL_TOLERANCE:
            break
        release = start
    return release


def contact_pixel(frames, landing):
    """where the ball touches the floor in the left image
    """
    before = [k for k in sorted(frames) if landing - 4 <= k < landing][-3:]
    after = [k for k in sorted(frames) if landing < k <= landing + 3][:2]
    if len(before) < 3 or len(after) < 2:
        return frames[landing]['uv'], 'lowest frame'
    t_fall = np.array([frames[k]['t'] for k in before])
    uv_fall = np.array([frames[k]['uv'] for k in before])
    t_bounce = np.array([frames[k]['t'] for k in after])
    v_bounce = np.array([frames[k]['uv'][1] for k in after])
    v_fall_fit = np.polyfit(t_fall, uv_fall[:, 1], 2)
    v_bounce_fit = np.polyfit(t_bounce, v_bounce, 1)
    meets = [r.real for r in np.roots(np.polysub(v_fall_fit, v_bounce_fit))
             if abs(r.imag) < 1e-9 and t_fall[-1] - 1e-3 <= r.real <= t_bounce[0] + 1e-3]
    if not meets:
        return frames[landing]['uv'], 'lowest frame'
    t_contact = min(meets)
    u = np.polyval(np.polyfit(t_fall, uv_fall[:, 0], 1), t_contact)
    v = np.polyval(v_fall_fit, t_contact)
    return np.array([u, v]), 'paths meet'


def pixel_to_floor(uv, calibration, height=BALL_RADIUS):
    """follow the camera ray through a left image pixel down to a height above the floor"""
    camera = calibration['camera']
    K = calibration['P1'][:, :3]
    ray = camera['rotation'] @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    s = (height - camera['position'][2]) / ray[2]
    return camera['position'] + s * ray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='grid_recordings')
    args = parser.parse_args()
    folder = Path(args.folder)
    videos = sorted(folder.glob('*.mp4'), key=video_number)

    # camera pose from the board
    plain = make_calibration((672, 376))
    up, height, spread, views = board_pose(videos, plain)
    save_camera_pose(folder, up, height,
                     note=f'from the floor checkerboard ({SQUARE_SIZE * 1000:.2f} mm squares), '
                          f'{views} views, height spread {spread * 1000:.1f} mm')
    print(f'camera pose from {views} views of the board: height {height:.4f} m '
          f'(spread {spread * 1000:.1f} mm), up in camera {np.round(up, 4)}')
    print(f'saved {folder / "camera_pose.json"}\n')

    rows = []
    print('video  release  landing  frames in air  contact from   landing x, y (m)   stereo says x, y')
    for video in videos:
        calibration = calibration_for(video)
        frames = track_ball(video, calibration)
        landing = find_landing_frame(frames)
        if landing is None:
            print(f'{video.name}: no landing found, skipped')
            continue
        release = find_release_frame(frames, landing)
        if release is None:
            print(f'{video.name}: no free fall found before the landing, skipped')
            continue
        uv, how = contact_pixel(frames, landing)
        spot = pixel_to_floor(uv, calibration)
        stereo = frames[landing]['world']
        print(f'{video_number(video):5d}  {release:7d}  {landing:7d}  {landing - release:13d}  '
              f'{how:<12}  ({spot[0]:5.2f}, {spot[1]:5.2f})     ({stereo[0]:5.2f}, {stereo[1]:5.2f})')
        rows.append([video.name, release, landing, f'{spot[0]:.3f}', f'{spot[1]:.3f}',
                     f'checkerboard floor plane, ball image position ({how})'])

    path = folder / 'ground_truth.csv'
    with open(path, 'w', newline='') as file:
        out = csv.writer(file)
        out.writerow(['video', 'release_frame', 'landing_frame', 'landing_x', 'landing_y', 'source'])
        out.writerows(rows)
    print(f'\nsaved {len(rows)} throws to {path}')


if __name__ == '__main__':
    main()
