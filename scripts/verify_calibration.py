"""Preview stereo ball detection and 3D camera coordinates; check a checkerboard on demand.

Run: python3 -m scripts.verify_calibration
Keys: b = ball detection on/off, c/h = color/Hough, v = check board,
r = raw/rectified preview, q = quit.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Support both `python -m scripts.verify_calibration` and direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ball_tracking_machine_vision.camera_capture import (
    open_camera,
    read_stereo_frame,
    split_side_by_side,
    build_rectification_maps,
    rectify,
    verify_calibration_checkerboard,
)


from scripts.checkerboard_pose import capture_world_pose
from ball_tracking_machine_vision.ball_detector import detect_ball
from ball_tracking_machine_vision.visualization import draw_detection, draw_predicted_path
from ball_tracking_machine_vision.kalman_filter import BallKalmanFilter, SIGMA_A
from ball_tracking_machine_vision.landing_predictor import find_landing_point, propagate_trajectory
from ball_tracking_machine_vision.simulation import BALL_RADIUS
from ball_tracking_machine_vision.triangulation import (
    compute_disparity, triangulate, camera_to_world,
)


def load_world_pose(path, image_size):
    """Read a measured pose from JSON, never assume the camera is at the origin.

    Required keys:
      source_frame: "rectified_left_camera"
      image_size: [width, height] for the matching rectification geometry
      R_camera_to_world: 3x3 rotation mapping camera axes into world axes
      camera_position_world: [x, y, z] in meters
    World origin and axes must be defined when measuring this pose. For the
    project's floor frame, world z points upward. Factory stereo R/T are NOT
    this pose. Remeasure the pose if the camera moves relative to the world.
    """
    with open(path) as file:
        data = json.load(file)
    if data.get('source_frame') != 'rectified_left_camera':
        raise ValueError('Pose source_frame must be rectified_left_camera.')
    if data.get('image_size') != list(image_size):
        raise ValueError('Pose image_size must match the current per-view resolution.')
    rotation = np.asarray(data['R_camera_to_world'], dtype=np.float64)
    translation = np.asarray(data['camera_position_world'], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError('Pose needs a 3x3 rotation and a 3-element position in meters.')
    if not (np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation))):
        raise ValueError('Pose values must be finite.')
    if not (np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError('Pose rotation must be orthonormal with determinant +1.')
    return rotation, translation


def world_position_label(position, pose):
    """Never relabel camera coordinates as world coordinates without a pose."""
    if pose is None:
        return 'World XYZ: pose not configured (--world-pose FILE)'
    if position is None:
        return 'World XYZ: unavailable (no valid stereo measurement)'
    world = camera_to_world(position, pose)
    if world is None or not np.all(np.isfinite(world)):
        return 'World XYZ: invalid transformation'
    x, y, z = world
    return f'World (m): x={x:.3f}  y={y:.3f}  z={z:.3f}'


def annotate_ball(image, result, side):
    """Draw on a copy, preserving the original image for other checks."""
    display = draw_detection(image, result)
    if result is None:
        label = f"{side}: no ball"
    else:
        u, v, radius = result
        label = f"{side}: u={u:.1f} v={v:.1f} radius={radius:.1f}px"
    cv2.putText(display, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)
    return display


class LivePrediction:
    """Maintain world-frame velocity estimates across frames of one throw."""

    def __init__(self, floor_height, ball_radius):
        self.floor_height = floor_height
        self.ball_radius = ball_radius
        self.reset()

    def reset(self):
        # Initial preview noise assumption: 3 cm per world axis. Tune with data.
        self.filter = BallKalmanFilter(SIGMA_A, np.eye(3) * 0.03**2)
        self.last_seen = None

    def step(self, measurement, timestamp):
        kf = self.filter
        # A long gap is a new track; never continue an old throw indefinitely.
        if self.last_seen is not None and timestamp - self.last_seen > 0.5:
            self.reset()
            kf = self.filter
        if kf.x is None:
            if measurement is None:
                return None, 'Prediction: waiting for stereo measurements'
            kf.initialize(np.asarray(measurement), timestamp)
            self.last_seen = timestamp
            return None, 'Prediction: collecting velocity measurements'
        if timestamp <= kf.last_time:
            return None, 'Prediction: waiting for increasing timestamps'
        if measurement is None:
            kf.predict(timestamp - kf.last_time)
            kf.last_time = timestamp
        else:
            if kf.step(np.asarray(measurement), timestamp):
                self.last_seen = timestamp
        if kf.n_updates < 3:
            return None, 'Prediction: collecting velocity measurements'
        state, covariance = kf.get_state()
        if not np.all(np.isfinite(state)) or state[2] <= self.floor_height + self.ball_radius:
            return None, 'Prediction: no airborne track'
        # The line needs only the nominal landing time, not Monte Carlo samples
        # from predict_landing(), which are for the landing uncertainty ellipse.
        landing = find_landing_point(state, self.floor_height, self.ball_radius)
        if landing is None or not np.isfinite(landing[1]) or landing[1] <= 0:
            return None, 'Prediction: no valid future landing'
        point, duration = landing
        future = propagate_trajectory(state, covariance, kf.sigma_a,
                                      min(duration, 5.0), dt=1 / 30)
        label = (f'Predicted landing (m): x={point[0]:.2f} y={point[1]:.2f}'
                 f' | in {duration:.2f}s | yellow: future path (max 5s)')
        if measurement is None:
            label += ' | coasting'
        return future['state'][:, :3], label


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, help='Raw stereo video filename in recordings/, or a video path')
    parser.add_argument('--loop', action='store_true', help='Restart recording and tracker at end of video')
    parser.add_argument('--device', default='/dev/video4', help='ZED video device')
    parser.add_argument('--width', type=int, default=672, help='Width of ONE view')
    parser.add_argument('--height', type=int, default=376, help='Height of ONE view')
    parser.add_argument('--fps', type=int, default=30, help='Requested capture FPS')
    # A 10x10-square board has 9x9 internal corners; 3.1 inches = 78.74 mm.
    parser.add_argument('--columns', type=int, default=9, help='Internal corner columns')
    parser.add_argument('--rows', type=int, default=9, help='Internal corner rows')
    parser.add_argument('--square-mm', type=float, default=78.74, help='Square width in mm')
    parser.add_argument('--method', choices=('color', 'hough'), default='color')
    parser.add_argument('--world-pose', type=Path, help='Measured camera-to-world pose JSON file')
    parser.add_argument('--floor-height', type=float, default=0.0)
    parser.add_argument('--ball-radius', type=float, default=BALL_RADIUS, help='Actual ball radius in meters')
    parser.add_argument('--save-world-pose', type=Path, default=PROJECT_ROOT / 'recordings/camera_pose.json', help='Pose output after pressing m and accepting the axes')
    args = parser.parse_args(argv)
    args.save_world_pose = args.save_world_pose.expanduser().resolve()
    if not np.isfinite(args.floor_height) or not np.isfinite(args.ball_radius) or args.ball_radius <= 0:
        parser.error('Floor height must be finite and ball radius positive.')
    if args.recording is not None:
        # Prefer an explicit existing path; otherwise search project recordings/.
        candidate = args.recording.expanduser()
        if not candidate.is_file():
            candidate = PROJECT_ROOT / 'recordings' / args.recording
        if not candidate.is_file():
            parser.error(f'Recording not found: {args.recording}')
        args.recording = candidate.resolve()
    if args.world_pose is not None:
        args.world_pose = args.world_pose.resolve()
    if min(args.width, args.height, args.fps) <= 0:
        parser.error('Dimensions and FPS must be positive.')
    if min(args.columns, args.rows) < 3:
        parser.error('Use at least 3 internal corners per direction.')
    if not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error('Square width must be positive and finite.')
    return args


def show_pair(window, left, right, label, guides=False, position_label=None, world_label=None, prediction_label=None):
    """Resize only the display; measurements always use full-resolution images."""
    preview = np.hstack((left, right))
    if guides:
        # Matching points should lie along the same horizontal guide line.
        for y in range(60, preview.shape[0], 60):
            cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 255, 0), 1)
    scale = min(1.0, 1400 / preview.shape[1])
    if scale < 1:
        preview = cv2.resize(preview, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
    # Add a separate status panel below the images; never cover image pixels.
    lines = [line for line in (label, position_label, world_label, prediction_label)
             if line is not None]
    if lines:
        image_height = preview.shape[0]
        preview = cv2.copyMakeBorder(preview, 0, 30 * len(lines) + 8, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for index, line in enumerate(lines):
            cv2.putText(preview, line, (10, image_height + 23 + 30 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow(window, preview)


def print_report(report):
    print('\n' + report['message'])
    print(f"Board found: left={report['found_left']}, right={report['found_right']}")
    if report['success']:
        print(f"Mean vertical mismatch: {report['mean_vertical_error_px']:.3f} px")
        print(f"Maximum vertical mismatch: {report['max_vertical_error_px']:.3f} px")
        print(f"Expected spacing: {1000 * report['expected_spacing_m']:.3f} mm")
        print(f"Measured mean spacing: {1000 * report['mean_spacing_m']:.3f} mm")
        print(f"Mean absolute spacing error: {report['mean_absolute_spacing_error_percent']:.2f}%")
        print('These are measurements, not an automatic pass. Inspect corner ordering.')


class RecordingSource:
    """Read raw side-by-side frames with video time, independent of processing speed."""

    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        self.index = 0
        self.previous_time = None
        try:
            if not self.capture.isOpened():
                raise RuntimeError(f'Could not open recording: {path}')
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError('Recording has no valid FPS for playback timing.')
            ok, self.first = self.capture.read()
            if not ok:
                raise ValueError('Recording contains no readable frames.')
            height, width = self.first.shape[:2]
            if width % 2:
                raise ValueError('Stereo recording must have an even combined width.')
            self.image_size = (width // 2, height)
        except Exception:
            self.release()
            raise

    def read(self):
        if self.index == 0:
            frame = self.first
        else:
            ok, frame = self.capture.read()
            if not ok:
                return None  # Normal end of recording.
        # Prefer presentation time, which also supports variable-rate videos.
        timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if (not np.isfinite(timestamp) or timestamp < 0
                or (self.previous_time is not None and timestamp <= self.previous_time)):
            timestamp = (0.0 if self.previous_time is None
                         else self.previous_time + 1.0 / self.fps)
        self.previous_time = timestamp
        self.index += 1
        left, right = split_side_by_side(frame)
        return left, right, timestamp

    def release(self):
        self.capture.release()


def main(argv=None):
    args = parse_args(argv)
    # Existing build_rectification_maps uses a relative docs/... path.
    os.chdir(PROJECT_ROOT)
    recording = RecordingSource(args.recording) if args.recording else None
    image_size = recording.image_size if recording else (args.width, args.height)
    # Recording dimensions choose the calibration; do not resize the source.
    try:
        pose = load_world_pose(args.world_pose, image_size) if args.world_pose else None
        maps = build_rectification_maps(image_size)
        _, _, _, _, P1, P2, _ = maps
        camera = recording if recording else open_camera(
            args.device, (2 * args.width, args.height), args.fps)
    except Exception:
        if recording:
            recording.release()
        raise
    if recording:
        print(f'Playing {args.recording.name}: views {image_size}, {recording.fps:.2f} FPS')
        print('The video must be raw left|right from this calibrated camera.')
    playback_origin = None
    predictor = LivePrediction(args.floor_height, args.ball_radius)
    rectified_preview = True
    ball_detection = True
    method = args.method
    frames = 0
    started = time.monotonic()
    loop_fps = 0.0

    print(f'Board: {args.columns} x {args.rows} INTERNAL corners, {args.square_mm:g} mm squares.')
    print('Keys: b=balls on/off, c=color, h=Hough, v=board, r=raw/rectified, q=quit.')
    print('Uses current ball_detector.py settings. Measurements use full-size rectified images.')
    print('3D: rectified LEFT CAMERA frame, meters; x right, y down, z forward.')
    print('World pose loaded.' if pose is not None else 'World pose not configured; camera coordinates remain available.')
    print('m: map floor from checkerboard, n: reset throw. Yellow line: predicted path.')
    print('Checkerboard checking may briefly pause the live preview.')

    try:
        while True:
            if recording:
                pair = recording.read()
                if pair is None:
                    if not args.loop:
                        print('End of recording.')
                        break
                    # Reopen instead of seeking so decoder and timestamps reset.
                    recording.release()
                    recording = RecordingSource(args.recording)
                    camera = recording
                    predictor.reset()
                    playback_origin = None
                    frames = 0
                    started = time.monotonic()
                    loop_fps = 0.0
                    print('Restarting recording.')
                    continue
                left, right, timestamp = pair
                if playback_origin is None:
                    playback_origin = time.monotonic() - timestamp
            else:
                left, right, timestamp = read_stereo_frame(camera)
            expected_shape = (image_size[1], image_size[0])
            if left.shape[:2] != expected_shape or right.shape[:2] != expected_shape:
                raise RuntimeError(
                    f'Camera returned views {left.shape[:2]} and {right.shape[:2]} '
                    f'(height, width); expected {expected_shape}. '
                    'Choose a supported capture resolution before rectification.'
                )

            # Build maps once, but apply them to each new stereo pair.
            left_rect, right_rect = rectify(left, right, maps)

            left_ball = right_ball = None
            position = None
            ball_status = 'Left: off | Right: off'
            position_label = 'Camera XYZ: detection off'
            if ball_detection:
                # Both detections use the same settings and corrected geometry.
                left_ball = detect_ball(left_rect, method=method)
                right_ball = detect_ball(right_rect, method=method)
                ball_status = f'Left: {"detected" if left_ball is not None else "not detected"} | Right: {"detected" if right_ball is not None else "not detected"}'
                position_label = 'Camera XYZ: unavailable (need both detections)'
                if left_ball is not None and right_ball is not None:
                    vertical_error = abs(left_ball[1] - right_ball[1])
                    disparity = compute_disparity(left_ball, right_ball)
                    # ball_status += f' | dv={vertical_error:.1f}px du={disparity:.1f}px'
                    # Recompute each frame so lost detections never leave stale XYZ.
                    position = (triangulate(left_ball, right_ball, P1, P2)
                                if vertical_error <= 5.0 else None)
                    if position is None:
                        position_label = 'Camera XYZ: invalid reconstruction'
                    else:
                        x, y, z = position
                        position_label = (
                            f'Camera (m): x={x:.3f}  y={y:.3f}  z={z:.3f}'
                            # ' | x right, y down, z forward'
                        )
                    # These independent detections are not guaranteed to be the
                    # same object. dv is a diagnostic, not a calibration verdict.

            # Keep detection outlines without the per-eye coordinate labels.
            # Detections are in rectified coordinates, so draw only on those views.
            if rectified_preview:
                display_left = draw_detection(left_rect, left_ball)
                display_right = draw_detection(right_rect, right_ball)
            else:
                display_left, display_right = left, right
            trajectory = None
            prediction_label = 'Prediction: press m to map the floor checkerboard'
            if pose is not None and ball_detection:
                trajectory, prediction_label = predictor.step(camera_to_world(position, pose), timestamp)
            elif not ball_detection:
                predictor.reset()
                prediction_label = 'Prediction: ball detection off'
            if rectified_preview and trajectory is not None:
                display_left = draw_predicted_path(display_left, trajectory, P1, pose)
                display_right = draw_predicted_path(display_right, trajectory, P2, pose)
            elif trajectory is not None:
                prediction_label += ' | press r for rectified overlay'

            # This measures preview-loop speed, not the hardware exposure rate.
            frames += 1
            elapsed = time.monotonic() - started
            if elapsed >= 1.0:
                loop_fps = frames / elapsed
                frames = 0
                started = time.monotonic()
            mode = 'Rectified' if rectified_preview else 'Raw'
            # if recording:
            #     mode += f' | video {timestamp:.2f}s'
            show_pair('ZED camera test', display_left, display_right,
                      f'{mode} | loop {loop_fps:.1f} FPS | {ball_status}',
                      guides=False, position_label=position_label,
                      world_label=world_position_label(position, pose),
                      prediction_label=prediction_label)

            # Pace the display near the recorded rate; process every frame even
            # if detection is slower. Filter time always comes from the video.
            delay = 1
            if recording:
                remaining = playback_origin + timestamp + 1 / recording.fps - time.monotonic()
                delay = max(1, min(1000, round(remaining * 1000)))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                rectified_preview = not rectified_preview
            elif key == ord('b'):
                ball_detection = not ball_detection
            elif key == ord('n'):
                predictor.reset()
            elif key == ord('c'):
                predictor.reset()
                method = 'color'
            elif key == ord('h'):
                predictor.reset()
                method = 'hough'
            elif key == ord('m'):
                new_pose = capture_world_pose(left_rect, P1, (args.columns, args.rows),
                                              args.square_mm / 1000, args.save_world_pose)
                if new_pose is not None:
                    pose = new_pose
                    predictor.reset()
                    rectified_preview = True
                # Mapping pauses playback; restart its wall-clock pacing anchor.
                if recording:
                    playback_origin = time.monotonic() - timestamp
            elif key == ord('v'):
                # Pass RAW images: verification performs its own rectification.
                report = verify_calibration_checkerboard(
                    left, right, maps,
                    square_size=args.square_mm / 1000.0,
                    pattern_size=(args.columns, args.rows),
                )
                print_report(report)
                show_pair('Checkerboard snapshot', report['left_preview'],
                          report['right_preview'], 'Last check: inspect terminal and corner ordering')
    finally:
        # Release the device on normal exit, Ctrl+C, or an exception.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
