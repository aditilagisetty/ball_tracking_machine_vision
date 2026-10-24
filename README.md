# Ball Tracking Machine Vision

Track a tossed tennis ball in 3D with a ZED 2i stereo camera, estimate its state with a Kalman filter, and predict where it will land.

## Layout

| File | What it does |
|---|---|
| `camera_capture.py` | Read and rectify stereo frames |
| `ball_detector.py` | Find the ball in each image |
| `triangulation.py` | Stereo pair to 3D position |
| `kalman_filter.py` | Gravity-model Kalman filter |
| `landing_predictor.py` | Extrapolate to the floor |
| `baseline.py` | Polynomial-fit baseline |
| `simulation.py` | Simulated throws for testing |
| `evaluation.py` | Landing error vs. fraction of flight observed |
| `visualization.py` | Overlays and plots |
| `tracker_node.py` | ROS 2 node tying it together |
