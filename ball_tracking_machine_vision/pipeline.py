"""ties detection, triangulation, filtering, and prediction together for one throw

    1. rectify
    2. find the ball in both images
    3. triangulate (camera frame)
    4. camera_to_world (world frame, z up)
    5. noise matrix R for that spot
    6. kalman filter step
    7. landing guess
"""

import json
from pathlib import Path

import cv2
import numpy as np

from ball_tracking_machine_vision.ball_detector import detect_ball
from ball_tracking_machine_vision.camera_capture import (
    build_rectification_maps,
    rectify,
    split_side_by_side,
)
from ball_tracking_machine_vision.kalman_filter import SIGMA_A, BallKalmanFilter
from ball_tracking_machine_vision.landing_predictor import predict_landing
from ball_tracking_machine_vision.recording import load_throw
from ball_tracking_machine_vision.stereo_noise import (
    PIXEL_NOISE,
    make_camera,
    measurement_covariance,
)
from ball_tracking_machine_vision.triangulation import camera_to_world, triangulate

# after rectification the ball is on the same row in both images, so the two
# detections can only be a few pixels apart up and down. more than this means
# one side found something else
MAX_ROW_DIFFERENCE_PX = 6.0


# where the floor is, seen from the camera, for the recordings in recordings/
# found by scripts/estimate_floor.py: it fits a plane through the ball while it
# rolls on the floor after landing (152 positions, 4.5 cm rms off the plane)
#   FLOOR_UP_IN_CAMERA: the up direction in the camera frame (x right, y down,
#       z forward). almost exactly camera "up" (minus y), so the camera is
#       nearly level: about 1.3 degrees of pitch and 2.6 degrees of roll
#   CAMERA_HEIGHT: how far the camera is above the floor (m)
# rerun the script if the camera is moved
FLOOR_UP_IN_CAMERA = np.array([-0.045, -0.999, -0.023])
CAMERA_HEIGHT = 0.774


def floor_camera_pose(up_in_camera=FLOOR_UP_IN_CAMERA, camera_height=CAMERA_HEIGHT):
    """camera pose in a world frame built from the floor

    world frame:
        z: straight up (the floor normal)
        x: the direction the camera looks, flattened onto the floor
        y: to the left (so x, y, z is right handed)
        origin: on the floor right below the camera

    gives back (rotation, position) in the same form as camera_to_world wants:
    rotation turns camera frame directions into world directions
    """
    up = np.asarray(up_in_camera, dtype=float)
    up = up / np.linalg.norm(up)
    forward = np.array([0.0, 0.0, 1.0])               # camera z axis
    forward = forward - (forward @ up) * up            # flatten onto the floor
    forward = forward / np.linalg.norm(forward)
    left = np.cross(up, forward)

    # rows are the world axes written in camera coordinates, so this matrix
    # takes a camera frame vector and gives its world x, y, z
    rotation = np.vstack([forward, left, up])
    position = np.array([0.0, 0.0, camera_height])
    return rotation, position


def make_calibration(image_size, camera_pose=None, pixel_noise=PIXEL_NOISE):
    """everything the pipeline needs to know about the camera in one dict

    image_size: (width, height) of one eye, for example (672, 376) for VGA
    camera_pose: (rotation, position) of the camera in the world.
        if None, the floor based pose from floor_camera_pose is used

    focal length and baseline come from the rectified calibration (P1, P2)
    so the noise model always matches the resolution the video was recorded at
    """
    maps = build_rectification_maps(image_size)
    P1, P2 = maps[4], maps[5]
    focal_length_px = P1[0, 0]
    baseline = -P2[0, 3] / P2[0, 0]          # P2[0, 3] is -focal * baseline

    rotation, position = floor_camera_pose() if camera_pose is None else camera_pose
    camera = make_camera(focal_length_px=focal_length_px, baseline=baseline,
                         pixel_noise=pixel_noise)
    camera['rotation'] = np.asarray(rotation, dtype=float)
    camera['position'] = np.asarray(position, dtype=float)
    return {'maps': maps, 'P1': P1, 'P2': P2, 'camera': camera}


# a recordings folder can hold its own measured camera pose, for example from
# the floor checkerboard (scripts/grid_ground_truth.py). it overrides the
# floor based pose above for every video in that folder
CAMERA_POSE_FILE = 'camera_pose.json'


def save_camera_pose(folder, up_in_camera, camera_height, note=''):
    """write a measured camera pose into a recordings folder"""
    data = {'up_in_camera': [float(v) for v in up_in_camera],
            'camera_height': float(camera_height), 'note': note}
    with open(Path(folder) / CAMERA_POSE_FILE, 'w') as file:
        json.dump(data, file, indent=2)


def load_camera_pose(folder):
    """(rotation, position) from a folder's camera_pose.json, or None if it has none"""
    path = Path(folder) / CAMERA_POSE_FILE
    if not path.exists():
        return None
    with open(path) as file:
        data = json.load(file)
    return floor_camera_pose(np.array(data['up_in_camera']), data['camera_height'])


def calibration_for(throw):
    """make_calibration for a throw, using its folder's camera_pose.json if there is one

    throw: a video file or a throw folder. the pose file is looked for next to it
    """
    throw = Path(throw)
    return make_calibration(image_size_of(throw), camera_pose=load_camera_pose(throw.parent))


def video_frames(video_path, start_frame=0, end_frame=None):
    """yield (frame_number, left, right, timestamp) from a side by side video

    start_frame, end_frame: only frames in this range (end included)
    the time of each frame is the timestamp stored in the video file. the
    camera drops frames now and then (a 133 ms gap instead of 67 ms), so
    frame number / fps would give the filter wrong time steps. frame number / fps
    is only the fallback for a file with no timestamps.
    used for the single video files in recordings/ and grid_recordings/;
    throw folders go through throw_frames instead
    """
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f'could not open video: {video_path}')
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_number = start_frame
    last_time = None
    try:
        while end_frame is None or frame_number <= end_frame:
            ok, frame = video.read()
            if not ok:
                break
            t = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if last_time is not None and t <= last_time:
                t = frame_number / fps          # no usable timestamps in this file
            last_time = t
            left, right = split_side_by_side(frame)
            yield frame_number, left, right, t
            frame_number += 1
    finally:
        video.release()


def video_file_of(throw):
    """the video file of a throw"""
    throw = Path(throw)
    return throw / 'video.avi' if throw.is_dir() else throw


def throw_frames(throw, start_frame=0, end_frame=None):
    """yield (frame_number, left, right, timestamp) from either kind of recording

    throw: a throw folder saved by recording.record_throw (video.avi and
        timestamps.csv), read with recording.load_throw so each frame keeps its
        real capture time; or a single video file like the mp4s in recordings/,
        timed as frame number / fps
    start_frame, end_frame: only frames in this range (end included)
    """
    throw = Path(throw)
    if not throw.is_dir():
        yield from video_frames(throw, start_frame, end_frame)
        return

    frames = load_throw(throw)
    try:
        for frame_number, (left, right, t) in enumerate(frames):
            if end_frame is not None and frame_number > end_frame:
                break
            if frame_number >= start_frame:
                yield frame_number, left, right, t
    finally:
        frames.close()     # releases the video file when stopping early


def video_fps(throw):
    """nominal frames per second of a throws video for writing overlay videos"""
    video = cv2.VideoCapture(str(video_file_of(throw)))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def image_size_of(video_path):
    """(width, height) of one eye of a side by side video or of a throw folders video"""
    video = cv2.VideoCapture(str(video_file_of(video_path)))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2,
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video.release()
    return size


def find_ball_in_both(left, right, method):
    """find the ball in both rectified images, or None

    the two detections only count as the same ball if they are on about
    the same row and the left one is further right (positive disparity,
    so the ball is in front of the camera)
    """
    left_detection = detect_ball(left, method)
    right_detection = detect_ball(right, method)
    if left_detection is None or right_detection is None:
        return None
    same_row = abs(left_detection[1] - right_detection[1]) <= MAX_ROW_DIFFERENCE_PX
    in_front = left_detection[0] - right_detection[0] > 0
    if not (same_row and in_front):
        return None
    return left_detection, right_detection


def measure_frame(left, right, calibration, method='color'):
    """turn one stereo frame into a world position of the ball

    gives back a dict with
        'detection': (left (u, v, r), right (u, v, r)) or None
        'position': (3,) world position or None
        'covariance': (3, 3) noise matrix R for that position or None
    """
    left_rect, right_rect = rectify(left, right, calibration['maps'])
    result = {'detection': None, 'position': None, 'covariance': None}

    detection = find_ball_in_both(left_rect, right_rect, method)
    if detection is None:
        return result
    result['detection'] = detection

    point_camera = triangulate(detection[0], detection[1], calibration['P1'], calibration['P2'])
    if point_camera is None:
        return result

    camera = calibration['camera']
    position = np.array(camera_to_world(point_camera, (camera['rotation'], camera['position'])))
    result['position'] = position
    result['covariance'] = measurement_covariance(position, camera)
    return result


def process_frame(left, right, timestamp, kalman_filter, calibration, method='color'):
    """detect, triangulate, and filter a single stereo frame

    starts the filter on the first frame that has a measurement
    gives back the measure_frame dict plus 'accepted': True if the filter used
    the measurement, False if it was thrown out as too far off, None if there was none
    """
    result = measure_frame(left, right, calibration, method)
    result['accepted'] = None
    result['nis'] = None
    if result['position'] is None:
        return result

    # R is worked out at the filters predicted position (see BallKalmanFilter.step)
    camera = calibration['camera']
    noise = lambda position: measurement_covariance(position, camera)  # noqa: E731

    if kalman_filter.x is None:
        kalman_filter.initialize(result['position'], timestamp, R=noise)
        result['accepted'] = True
    else:
        result['accepted'] = kalman_filter.step(result['position'], timestamp, R=noise)
        result['nis'] = kalman_filter.last_nis
    return result


def track_throw(frames, calibration, floor_height=0.0, sigma_a=SIGMA_A,
                method='color', rng=None):
    """run the full pipeline on a throw

    frames: (frame_number, left, right, timestamp) for each frame,
        for example video_frames(video, release_frame, landing_frame)

    gives back a dict of lists, one entry per frame:
        'frame', 't', 'detection', 'position', 'covariance', 'accepted',
        'nis': how surprising each measurement was (averages about 3 if
            the noise settings are right), or None
        'estimate': filter state (6,) or None
        'prediction': predict_landing dict or None
    plus 'measurements': only the frames that have a position, as
        {'t', 'position', 'covariance'}, the format evaluation.py uses
    """
    rng = np.random.default_rng() if rng is None else rng
    kalman_filter = BallKalmanFilter(sigma_a, np.eye(3) * 0.01)   # every frame brings its own R
    keys = ['frame', 't', 'detection', 'position', 'covariance', 'accepted',
            'nis', 'estimate', 'prediction']
    track = {key: [] for key in keys}

    for frame_number, left, right, t in frames:
        result = process_frame(left, right, t, kalman_filter, calibration, method)

        estimate, prediction = None, None
        if kalman_filter.x is not None:
            estimate = kalman_filter.get_state()[0]
            if result['accepted']:
                prediction = predict_landing(kalman_filter, floor_height, rng=rng)

        track['frame'].append(frame_number)
        track['t'].append(t)
        track['estimate'].append(estimate)
        track['prediction'].append(prediction)
        for key in ['detection', 'position', 'covariance', 'accepted', 'nis']:
            track[key].append(result[key])

    measured = [i for i, p in enumerate(track['position']) if p is not None]
    track['measurements'] = {
        't': np.array([track['t'][i] for i in measured]),
        'position': np.array([track['position'][i] for i in measured]).reshape(-1, 3),
        'covariance': np.array([track['covariance'][i] for i in measured]).reshape(-1, 3, 3),
    }
    return track
