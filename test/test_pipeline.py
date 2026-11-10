"""pipeline tests on fake stereo videos made from a simulated throw

we draw a red ball where the camera would see it in the left and right
images so we know exactly where the ball really was
"""

import csv

import cv2
import numpy as np

from ball_tracking_machine_vision.camera_capture import build_rectification_maps
from ball_tracking_machine_vision.evaluation import evaluate_all
from ball_tracking_machine_vision.pipeline import throw_frames, track_throw, video_frames
from ball_tracking_machine_vision.simulation import BALL_RADIUS, simulate_throw
from ball_tracking_machine_vision.stereo_noise import make_camera

IMAGE_SIZE = (672, 376)      # VGA same as the recordings
RED_BGR = (0, 0, 180)
GRAY_BGR = (128, 128, 128)


def fake_calibration():
    """real VGA calibration, but with rectification turned off (identity maps)
    so the images we draw are already rectified"""
    real = build_rectification_maps(IMAGE_SIZE)
    P1, P2 = real[4], real[5]
    width, height = IMAGE_SIZE
    map_x, map_y = np.meshgrid(np.arange(width, dtype=np.float32),
                               np.arange(height, dtype=np.float32))
    maps = (map_x, map_y, map_x, map_y, P1, P2, real[6])
    camera = make_camera(focal_length_px=P1[0, 0], baseline=-P2[0, 3] / P2[0, 0])
    return {'maps': maps, 'P1': P1, 'P2': P2, 'camera': camera}


def draw_frame(position, calibration):
    """side by side image of the ball at this world position"""
    camera = calibration['camera']
    point_camera = camera['rotation'].T @ (position - camera['position'])
    images = []
    for P in (calibration['P1'], calibration['P2']):
        image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), GRAY_BGR, dtype=np.uint8)
        u, v, w = P @ np.append(point_camera, 1.0)
        radius = calibration['P1'][0, 0] * BALL_RADIUS / point_camera[2]
        # shift=4 draws with 1/16 pixel precision so the center is not rounded
        center = (round(u / w * 16), round(v / w * 16))
        cv2.circle(image, center, round(max(2.0, radius) * 16), RED_BGR, -1,
                   lineType=cv2.LINE_AA, shift=4)
        images.append(image)
    return np.hstack(images)


def fake_throw():
    return simulate_throw([0.5, 0.0, 1.2], [2.0, 0.3, 2.5], dt=1 / 60)


def test_pipeline_finds_ball_in_3d():
    calibration = fake_calibration()
    throw = fake_throw()
    frames = []
    for i, (t, position) in enumerate(zip(throw['t'], throw['position'])):
        image = draw_frame(position, calibration)
        frames.append((i, image[:, :IMAGE_SIZE[0]], image[:, IMAGE_SIZE[0]:], t))

    track = track_throw(frames, calibration, rng=np.random.default_rng(0))
    meas = track['measurements']

    assert len(meas['t']) == len(throw['t'])                 # found in every frame
    errors = np.linalg.norm(meas['position'] - throw['position'], axis=1)
    assert np.median(errors) < 0.05                          # within 5 cm (pixel rounding)
    final = [p for p in track['prediction'] if p is not None][-1]['point']
    assert np.linalg.norm(final[:2] - throw['landing_point'][:2]) < 0.10


def test_evaluate_all_on_fake_video(tmp_path):
    calibration = fake_calibration()
    throw = fake_throw()
    fps = 60

    # write the fake throw as a video plus one frame on the floor at the end
    video_path = tmp_path / 'fakethrow.avi'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), fps,
                             (2 * IMAGE_SIZE[0], IMAGE_SIZE[1]))
    for position in throw['position']:
        writer.write(draw_frame(position, calibration))
    landing_frame = len(throw['position'])
    writer.write(draw_frame(throw['landing_point'] + [0, 0, BALL_RADIUS], calibration))
    writer.release()

    assert len(list(video_frames(video_path))) == landing_frame + 1

    truth_path = tmp_path / 'ground_truth.csv'
    with open(truth_path, 'w', newline='') as file:
        out = csv.writer(file)
        out.writerow(['video', 'release_frame', 'landing_frame', 'landing_x', 'landing_y'])
        out.writerow(['fakethrow.avi', 0, landing_frame, *throw['landing_point'][:2]])

    table, rows = evaluate_all(tmp_path, truth_path, calibration=calibration,
                               fractions=[0.5, 0.9])
    assert len(rows) == 1
    assert table[-1]['filter_median'] < 0.10      # within 10 cm after 90% of the flight


def test_throw_folder_from_recording(tmp_path):
    """a throw folder in recording.pys format runs
    through evaluate_all using the saved timestamps instead of frame / fps"""
    calibration = fake_calibration()
    throw = fake_throw()

    folder = tmp_path / 'throw_01'
    folder.mkdir()
    writer = cv2.VideoWriter(str(folder / 'video.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 60,
                             (2 * IMAGE_SIZE[0], IMAGE_SIZE[1]))
    for position in throw['position']:
        writer.write(draw_frame(position, calibration))
    landing_frame = len(throw['position'])
    writer.write(draw_frame(throw['landing_point'] + [0, 0, BALL_RADIUS], calibration))
    writer.release()

    # uneven capture times, like a real camera: the pipeline must use these
    times = np.append(throw['t'], throw['landing_time']) + 5.0
    with open(folder / 'timestamps.csv', 'w', newline='') as file:
        out = csv.writer(file)
        out.writerow(['frame_index', 'timestamp'])
        for i, t in enumerate(times):
            out.writerow([i, f'{t:.6f}'])

    frames = list(throw_frames(folder, 2, 4))
    assert [f[0] for f in frames] == [2, 3, 4]
    assert np.allclose([f[3] for f in frames], times[2:5])

    truth_path = tmp_path / 'ground_truth.csv'
    with open(truth_path, 'w', newline='') as file:
        out = csv.writer(file)
        out.writerow(['video', 'release_frame', 'landing_frame', 'landing_x', 'landing_y'])
        out.writerow(['throw_01', 0, landing_frame, *throw['landing_point'][:2]])

    table, rows = evaluate_all(tmp_path, truth_path, calibration=calibration,
                               fractions=[0.5, 0.9])
    assert len(rows) == 1
    assert table[-1]['filter_median'] < 0.10
