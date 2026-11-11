## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Recorded Ball Tracking

Open a terminal in the project’s root folder and run:

```bash
env -u GTK_PATH python3 -m scripts.run_throw "grid_recordings/video (2).mp4" --start 30 --end 36
```

This processes frames 30–36, covering the ball’s flight. No camera connection is required.

### Outputs

Open `output/video (2)/` to view:

- `overlay.mp4` — recorded footage with ball detection and predicted trajectory.
- `track_3d.png` — measured and filtered positions with the final predicted landing point.
- `landing_uncertainty.png` — predicted landing position and its uncertainty ellipse.

Running the same recording again overwrites these output files.

### Use Another Recording

Find the recording in `grid_recordings/ground_truth.csv`, then replace:

- The video filename.
- `--start` with its `release_frame`.
- `--end` with its `landing_frame`.

Use the flight range rather than the entire recording, since the motion model does not handle the ball being held, bouncing, or rolling.