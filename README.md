# Ball Tracking Machine Vision

Track a tossed tennis ball in 3D with a ZED 2i stereo camera, estimate its
state with a Kalman filter, and predict where it will land.

Plain Python + OpenCV, no ROS. The ZED 2i is used as a USB camera: one
side-by-side image per frame, which we split, rectify, and triangulate ourselves.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

| File | What it does |
|---|---|
| `camera_capture.py` | Open the ZED, split and rectify stereo frames |
| `recording.py` | Save/load recorded throws (video + timestamps) |
| `ball_detector.py` | Find the ball: color threshold and Hough circles |
| `triangulation.py` | Stereo pair to 3D position |
| `kalman_filter.py` | Gravity-model Kalman filter |
| `landing_predictor.py` | Extrapolate to the floor |
| `baseline.py` | Polynomial-fit baseline |
| `simulation.py` | Simulated throws for testing the filter |
| `pipeline.py` | Runs every stage on one throw |
| `evaluation.py` | Landing error vs. fraction of flight observed |
| `visualization.py` | Overlays and plots |

Scripts: `record.py`, `tune_detector.py`, `run_throw.py`, `simulate_filter.py`,
`run_evaluation.py`.

## Priorities and backup plans

| Priority | Component | Backup if it stalls |
|---|---|---|
| 1 | Kalman filter (gravity) | `filterpy` for the filter itself, keep our own model |
| 1 | Landing prediction + error plot | none, this is the result |
| 2 | Stereo triangulation | `cv2.triangulatePoints`, or ZED SDK depth |
| 2 | Ball detection | color threshold only; better lighting/higher fps |
| 3 | Visualizations | simple matplotlib plots only |
| Stretch | Drag, bounce model in the filter, goal prediction | |

## Data format between stages

The hand-off between perception and estimation is a list of
`(timestamp, x, y, z)` in a world frame with **z up, origin on the floor**,
in meters and seconds.
