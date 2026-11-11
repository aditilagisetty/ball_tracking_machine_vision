"""Save raw stereo throws and replay them with their measured frame times.

Each throw is a folder containing video.avi (left | right BGR frames) and
 timestamps.csv (frame_index,timestamp). Timestamps are seconds since the
first saved frame, sampled after camera.read(); they are approximate host
read times, not hardware exposure timestamps. AVI's nominal FPS does not
replace those timestamps when estimating velocity.

---------------------------AI DISCLAIMER------------------------------------
These functions are primarily AI generated, developing this file ourselves is
out of scope for this project (and does not contribute to our learning goals)
"""

import csv
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from ball_tracking_machine_vision.camera_capture import split_side_by_side


def record_throw(camera, output_dir, duration):
    """Record raw stereo frames for a duration and return the output Path.

    Args:
        camera: An already-open OpenCV VideoCapture providing uint8 BGR stereo
            frames. The caller remains responsible for releasing this camera.
        output_dir: New or existing folder; existing recording files are never
            overwritten. Incomplete capture files are discarded on failure.
        duration: Positive finite recording duration in seconds. A blocking
            camera read may make the function take longer than this duration.

    Returns:
        pathlib.Path: Folder containing video.avi and timestamps.csv.

    Raises:
        ValueError: Invalid duration, FPS, or frame format.
        RuntimeError: Camera read/encoder failure or no captured frames.
        FileExistsError: Either destination recording file already exists.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive and finite')
    if not camera.isOpened():
        raise RuntimeError('Camera is not open')
    fps = float(camera.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Camera must report a valid nominal FPS')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / 'video.avi'
    time_path = output_dir / 'timestamps.csv'
    if video_path.exists() or time_path.exists():
        raise FileExistsError(f'Recording already exists in {output_dir}')

    # Work in a temporary subfolder so failed captures are not listed as throws.
    with tempfile.TemporaryDirectory(prefix='.capture-', dir=output_dir) as scratch:
        temp_video = Path(scratch) / 'video.avi'
        temp_times = Path(scratch) / 'timestamps.csv'
        writer = None
        count = 0
        first_time = None
        started = time.monotonic()
        try:
            with temp_times.open('w', newline='') as file:
                rows = csv.writer(file)
                rows.writerow(['frame_index', 'timestamp'])
                while time.monotonic() - started < duration:
                    ok, frame = camera.read()
                    read_time = time.monotonic()
                    if not ok or frame is None:
                        raise RuntimeError('Camera stopped providing frames')
                    if (frame.dtype != np.uint8 or frame.ndim != 3
                            or frame.shape[2] != 3):
                        raise ValueError('Expected uint8 BGR stereo frames')
                    height, width = frame.shape[:2]
                    # Even height also avoids silent cropping by common encoders.
                    if width < 2 or height < 2 or width % 2 or height % 2:
                        raise ValueError('Stereo width and video height must be positive even dimensions')
                    if writer is None:
                        size = (width, height)
                        writer = cv2.VideoWriter(str(temp_video),
                                                cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
                        if not writer.isOpened():
                            raise RuntimeError('Could not create MJPG video writer')
                        first_time = read_time
                    elif (width, height) != size:
                        raise RuntimeError('Camera resolution changed during recording')
                    # Store the full unrectified stereo image for reproducible tests.
                    writer.write(frame)
                    rows.writerow([count, read_time - first_time])
                    count += 1
        finally:
            if writer is not None:
                writer.release()
        if count == 0:
            raise RuntimeError('No frames were recorded')

        # Check the encoder really saved one readable frame per timestamp.
        check = cv2.VideoCapture(str(temp_video))
        saved = 0
        try:
            while True:
                ok, frame = check.read()
                if not ok:
                    break
                if frame.shape[:2] != (size[1], size[0]):
                    raise RuntimeError('Encoded frame size differs from captured size')
                saved += 1
        finally:
            check.release()
        if saved != count:
            raise RuntimeError(f'Encoder saved {saved} frames; expected {count}')

        # Hard links publish completed files without overwriting concurrent output.
        video_path.hardlink_to(temp_video)
        try:
            time_path.hardlink_to(temp_times)
        except BaseException:
            video_path.unlink()
            raise
    return output_dir


def load_throw(throw_dir):
    """Yield (left, right, timestamp) from a recorded throw folder.

    Images are raw BGR NumPy arrays, not rectified. Timestamps are elapsed
    seconds from capture, independent of replay/processing speed. This generator
    does not sleep; the caller controls display timing. If stopping early, call
    the generator's close() method to release its video handle promptly.

    Raises FileNotFoundError for missing files, ValueError for malformed timing
    data, and RuntimeError for unreadable video or mismatched frame counts.
    """
    folder = Path(throw_dir)
    video_path, time_path = folder / 'video.avi', folder / 'timestamps.csv'
    if not video_path.is_file() or not time_path.is_file():
        raise FileNotFoundError(f'Expected video.avi and timestamps.csv in {folder}')
    timestamps = []
    with time_path.open(newline='') as file:
        rows = csv.DictReader(file)
        if rows.fieldnames != ['frame_index', 'timestamp']:
            raise ValueError('Expected CSV columns: frame_index,timestamp')
        for index, row in enumerate(rows):
            try:
                frame_index = int(row['frame_index'])
                timestamp = float(row['timestamp'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid timestamp CSV row {index + 2}') from exc
            if frame_index != index:
                raise ValueError('Frame indices must be consecutive, starting at zero')
            if not np.isfinite(timestamp) or timestamp < 0:
                raise ValueError('Timestamps must be finite and nonnegative')
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError('Timestamps must strictly increase')
            timestamps.append(timestamp)
    if not timestamps:
        raise ValueError('Recording has no timestamps')

    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise RuntimeError(f'Could not open {video_path}')
        shape = None
        for timestamp in timestamps:
            ok, frame = video.read()
            if not ok:
                raise RuntimeError('Video has fewer readable frames than timestamps')
            if frame.shape[1] % 2:
                raise ValueError('Stereo frame width must be even')
            if shape is None:
                shape = frame.shape
            elif frame.shape != shape:
                raise ValueError('Video frame dimensions changed')
            left, right = split_side_by_side(frame)
            yield left, right, timestamp
        if video.read()[0]:
            raise RuntimeError('Video has more frames than timestamps')
    finally:
        video.release()


def list_throws(data_dir):
    """Return sorted Paths to immediate subfolders with both recording files.

    Missing directories return an empty list. A non-directory raises
    NotADirectoryError. Standalone MP4 files are not part of this folder format.
    File contents are validated by load_throw(), not by this listing function.
    """
    folder = Path(data_dir)
    if not folder.exists():
        return []
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_dir() and not p.name.startswith('.capture-')
                  and (p / 'video.avi').is_file()
                  and (p / 'timestamps.csv').is_file())
