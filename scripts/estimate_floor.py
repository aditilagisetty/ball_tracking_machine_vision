"""find where the floor is from the ball rolling on it, which gives the camera pose
"""

import numpy as np

from ball_tracking_machine_vision.camera_capture import rectify
from ball_tracking_machine_vision.pipeline import (
    find_ball_in_both,
    image_size_of,
    make_calibration,
    video_frames,
)
from ball_tracking_machine_vision.simulation import BALL_RADIUS
from ball_tracking_machine_vision.triangulation import triangulate

# frames where the ball is rolling on the floor (after the bounces), per video
ROLLING_FRAMES = {
    'recordings/goodvideo1.mp4': (60, 88),
    'recordings/goodvideo3.mp4': (62, 84),
    'recordings/goodvideo4.mp4': (40, 72),
    'recordings/goodvideo5.mp4': (80, 104),
    'recordings/goodvideo6.mp4': (86, 127),
}


def rolling_points(video, start, end, calibration):
    """3D ball positions in the camera frame while it rolls"""
    points = []
    for _, left, right, _ in video_frames(video, start, end):
        left_rect, right_rect = rectify(left, right, calibration['maps'])
        detection = find_ball_in_both(left_rect, right_rect, 'color')
        if detection is None:
            continue
        point = triangulate(detection[0], detection[1], calibration['P1'], calibration['P2'])
        if point is not None:
            points.append(point)
    return np.array(points)


def fit_plane(points):
    """best plane through the points: (center, unit normal, rms distance off it)

    the normal is the direction the points spread the least in, which is the
    last right singular vector of the centered points
    """
    center = points.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(points - center)
    normal = vt[2]
    rms = singular_values[2] / np.sqrt(len(points))
    return center, normal, rms


def main():
    first_video = next(iter(ROLLING_FRAMES))
    calibration = make_calibration(image_size_of(first_video))

    all_points = []
    for video, (start, end) in ROLLING_FRAMES.items():
        points = rolling_points(video, start, end, calibration)
        print(f'{video}: {len(points)} rolling positions')
        all_points.append(points)
    points = np.vstack(all_points)

    center, normal, rms = fit_plane(points)
    if normal[1] > 0:
        normal = -normal       # camera y points down so up has negative y

    # distance from the camera to the plane of ball centers
    height_of_centers = normal @ (np.zeros(3) - center)
    camera_height = height_of_centers + BALL_RADIUS

    tilt = np.degrees(np.arcsin(-normal[2]))     # + means looking down
    roll = np.degrees(np.arcsin(normal[0]))
    print(f'\n{len(points)} positions, {rms * 100:.1f} cm rms off the plane')
    print(f'camera pitch about {tilt:.1f} deg, roll about {roll:.1f} deg')
    print('\nput these in pipeline.py:')
    print(f'FLOOR_UP_IN_CAMERA = np.array([{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}])')
    print(f'CAMERA_HEIGHT = {camera_height:.3f}')


if __name__ == '__main__':
    main()
