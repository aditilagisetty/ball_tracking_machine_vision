These instructions are for Linux / MacOS devices
## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Recorded Ball Tracking

Open a terminal in the project’s root folder and run:

```bash
python3 -m scripts.run_throw "grid_recordings/video (2).mp4" --start 30 --end 36
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

## Run Simulation and Evaluation

Run these commands from the project’s root folder after completing setup.

### Evaluate Simulated Throws

```bash
python3 -m scripts.run_evaluation
```

This evaluates the Kalman filter and polynomial baseline on 200 simulated throws with measurement noise and dropped frames. It prints prediction errors and saves:

- `output/error_vs_fraction_sim.png`

### Evaluate Recorded Throws

```bash
python3 -m scripts.run_evaluation --real --videos grid_recordings --truth grid_recordings/ground_truth.csv
```

This evaluates the recorded grid throws using the flight ranges and reference landing positions in `ground_truth.csv`. It prints prediction errors and saves:

- `output/error_vs_fraction_real.png`

### Test the Filter on Simulated Motion

```bash
python3 -m scripts.simulate_filter
```

This runs a simulated-throw demonstration and compares filter uncertainty against actual estimation errors using NEES. It prints results for different process-noise and drag settings.

### Run Automated Tests

```bash
python3 -m pytest -q
```

This runs the automated tests. The current repository contains 21 tests. If ros is installed and you are getting a yaml error, run this instead:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

No camera connection is required for these commands. Rerunning evaluation overwrites the corresponding output graph.
