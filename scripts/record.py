"""record throws from the ZED 2i into data/ one folder per throw

    PYTHONPATH=. .venv/bin/python scripts/record.py --duration 4

press enter to record each throw or type q and enter to stop. each throw is
saved as data/throw_01/, data/throw_02/, etc. with the raw side by side video
and the capture time of every frame 

then run a throw through the pipeline with:
    PYTHONPATH=. .venv/bin/python scripts/run_throw.py data/throw_01
"""

import argparse
from pathlib import Path

from ball_tracking_machine_vision.camera_capture import open_camera
from ball_tracking_machine_vision.recording import record_throw


def next_throw_dir(data_dir):
    """the first unused folder name: throw_01, throw_02, ..."""
    n = 1
    while (Path(data_dir) / f'throw_{n:02d}').exists():
        n += 1
    return Path(data_dir) / f'throw_{n:02d}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/dev/video4', help='the ZED 2i video device')
    parser.add_argument('--width', type=int, default=1344, help='both eyes side by side (VGA)')
    parser.add_argument('--height', type=int, default=376)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--duration', type=float, default=4.0, help='seconds per throw')
    parser.add_argument('--out', default='data', help='folder the throw folders go in')
    args = parser.parse_args()

    camera = open_camera(args.source, (args.width, args.height), args.fps)
    try:
        while True:
            folder = next_throw_dir(args.out)
            answer = input(f'enter to record {folder.name} ({args.duration:g} s), q to quit: ')
            if answer.strip().lower() == 'q':
                break
            print('recording...')
            record_throw(camera, folder, args.duration)
            print(f'saved {folder}')
    finally:
        camera.release()


if __name__ == '__main__':
    main()
