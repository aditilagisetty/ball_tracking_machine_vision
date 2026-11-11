# Ball Tracking and Trajectory Prediction

*By: Aditi Lagisetty and Matthew Townsend*
## Run Instructions
To run our recorded ball tracking, refer to the [run_instructions.md](https://github.com/aditilagisetty/ball_tracking_machine_vision/blob/main/run_instructions.md) file

## Project Goal and Context

Our goal was to build a vision system that uses a ZED 2i stereo camera and a Kalman filter to track a thrown ball in 3D and predict where it will land, answering how early and how confidently that prediction can be made.

## Methods and Implementation

### Overview

To track the ball and predict where it would land, we created a system that takes images from the stereo camera and processes them through several steps. First, we capture the left and right camera images and correct them using the camera’s calibration values. We then use our chosen ball detection method to find the ball’s center in each image, and use triangulation to calculate its three-dimensional position. These positions are converted into world coordinates relative to our chosen origin on the floor.

Next, we use a Kalman filter to account for noise in the measurements and estimate the ball’s position and velocity over time. These estimates are used to predict the ball’s future trajectory and landing position.

Finally, we display the ball detections and predicted trajectory on the camera images so we can visually check how well the system is tracking the ball.

These concepts were the main focus of our project. Many of our scripts for visual verification were not completely hand-made by us as developing them ourselves was not in scope for the project. The bulk of this section will go over what we implemented by hand and briefly describe what the external helper scripts do and where we got them.

### Ball Detection — ball_detector.py

The ball detection module works to detect the ball using the stereo camera. To approach camera ball detection, we tried two different methods.

First, we implemented ball detection through color tracking. We created the make_color_mask function that converts colors of an image to HSV values, and then takes a high and low HSV range to create a color mask for the image. It returns an array where pixels that are within the color range are set to 255, and makes pixels that are not set to 0. Then we created the detect_ball_color function that finds boundary lines of the area in the array with the most amount of matching colors- or the largest color blob. The function then returns the center coordinates and the radius of the circle.

We also implemented ball detection through using a hough transform in the function detect_ball_hough. We first converted the image to grayscale, then blurred it to prep it for the hough transform. We applied the transform using opencv’s hough circles function, using specific parameters to capture the circular shape of the ball. This function also returned the center coordinates and radius of the circle.

Finally, this module contains two functions calling the ball detection – one for detecting the ball in one camera, and another for detecting the ball on both sides of the stereo camera. The single camera detector was strictly used for testing with the stereo camera ball detector being used in our final simulation.

### Color vs Hough Detection Design Decision

After using a visualizer to compare how both methods tracked our ball, we decided to use color tracking. Color tracking was more consistent with the hough detection method, which ended up being all over the place. The hough circles method struggled detecting the ball even while it was stationary, so we decided not to use this method as trajectory prediction would involve the ball moving at a higher speed. Color detection was successful at tracking the ball while it was stationary and while it was moving, making it our method of choice for ball detection.


### Camera Capture — camera_capture.py

The camera capture module works to capture images from the stereo camera and prepare them for ball detection and triangulation. First, we created the open_camera function to connect to the camera and set the resolution and frame rate. Since the stereo camera returns the left and right images side by side in one frame, we split the frame into two separate images and return them with a timestamp. We then created the load_calibration function to read the camera’s factory calibration file, which contains the focal lengths, optical centers, lens distortion values, and the position and rotation between the two cameras. These values are used in build_rectification_maps to calculate how each image needs to be corrected. Finally, the rectify function applies these maps to reduce lens distortion and align matching points in the left and right images along the same horizontal row. This prepares the images for triangulation, where the difference in the ball’s horizontal position between the two images is used to calculate its distance from the camera.

### Triangulation and Coordinate Transformation — triangulation.py

The triangulation module works to turn the ball’s position in the two camera images into a 3D position in the camera and world frame. First, we created the compute_disparity function to calculate the difference between the ball’s horizontal pixel coordinates in the left and right images. This difference is called disparity, where a larger disparity generally means the ball is closer to the camera. We then created the triangulate function, which takes the ball’s center coordinates from both images and the projection matrices created during camera rectification. Using OpenCV’s triangulatePoints function, we calculate the ball’s 3D position relative to the camera. Finally, the camera_to_world function uses the camera’s known position and orientation to rotate and shift these coordinates into our world frame. This allows us to track the ball relative to our chosen origin on the floor and use those positions for trajectory prediction.

### Motion Estimation — kalman_filter.py

The Kalman filter module smooths out the noisy 3D positions from the camera and estimates the ball's velocity, which the camera can't measure directly. The filter keeps track of six numbers: the ball's position (x, y, z) and its velocity (vx, vy, vz). Each new frame goes through two steps. In the predict step, we move the ball forward in time using basic physics: position changes by velocity times the time step, and gravity pulls the z velocity down. In the update step, we compare that prediction to the new camera measurement and blend the two. How much we trust each one depends on how uncertain the prediction is and how noisy the measurement is. The filter has two kinds of noise to keep track of. The first is process noise (SIGMA_A), which covers forces our model doesn't include, like air drag and spin. We chose its value of 1.85 m/s² by running the filter on simulated throws and checking that its uncertainty matched its actual errors (a check called NEES). The second is measurement noise (R), which comes from the stereo noise model described below. R changes from frame to frame because the camera gets less accurate as the ball moves farther away.

We also added protection against bad detections. Every measurement gets a surprise score called NIS, which says how far it is from where the filter expected it, scaled by the expected uncertainty. If the score is above 11.34 (the 99% cutoff), we throw the measurement out because it is probably a false detection. We don't do this check for the first 3 updates, since the velocity guess is still poor then. If 5 measurements in a row get thrown out, we assume the filter has gone wrong rather than the camera, and we restart it from the latest measurement.

### Trajectory and Landing Prediction — landing_predictor.py

The landing predictor takes the filter's current estimate and works out where and when the ball will hit the floor. The filter itself only models gravity. Our ball is small and light, though, so air drag slows it down a lot over a whole flight. With drag left out, the predictions landed too far away. To fix this, the predictor flies the ball forward in many tiny 1 ms steps, using both gravity and quadratic drag where drag acceleration = −k·|v|·v. It stops when the ball's center is one ball radius above the floor, which is the moment the bottom of the ball touches down. Next, we estimate how sure we are about the landing spot. We carry the filter's uncertainty forward to the landing time, draw 500 random possible ball states from that uncertainty, and fly each one to the floor. The spread of those 500 landing points gives a covariance, which we draw as a 95% ellipse around the predicted landing point.

### Visualization — visualization.py

For visualization, our visualization.py file has two purposes. First, it draws the detection overlay for the ball, creating a circle around the center coordinates of what it detects to be the ball. It also draws a red dot at the center so we can visually check how accurately the detector is tracking the ball. Along with the detection overlay, we draw the predicted trajectory onto the camera image. To do this, we convert the predicted world positions back into camera coordinates and then use the projection matrix to get their pixel coordinates. We connect these points with yellow lines to show the path the ball is predicted to follow.

The second purpose of this file is to create plots for analyzing the tracking and prediction results. We use a three-dimensional plot to compare the measured ball positions, the Kalman filter’s estimated positions, and the predicted path or landing point. We also plot the predicted landing position with an ellipse showing its uncertainty, where a larger ellipse means the prediction is less certain. Finally, we plot the landing prediction error against how much of the ball’s flight has been observed, comparing the Kalman filter with a polynomial baseline. These plots help us evaluate how well the system tracks the ball and how early it can accurately predict where the ball will land.

### Tracking Pipeline — pipeline.py

The pipeline connects all the pieces so that one function can process a whole throw. For each stereo frame it rectifies the left and right images, finds the ball in both, triangulates the ball's 3D position in the camera frame, converts it to the world frame, works out the measurement noise R for that spot, runs a Kalman filter step, and makes a new landing prediction.

### Stereo Measurement Uncertainty — stereo_noise.py

stereo_noise.py estimates how noisy each 3D measurement is, using the camera's geometry. Because depth = baseline × focal length / disparity, a small pixel error in disparity turns into a depth error that grows with the square of the distance. Sideways error only grows in proportion to the distance. The camera is much better at telling where the ball is across the image than how far away it is.

This builds this noise as a simple diagonal matrix in the camera frame, then rotates it into the world frame to get the R matrix the filter uses. Both the simulation and the real pipeline use the same functions, so the simulated noise matches what the real camera produces.

### Simulation — simulation.py

The simulation module creates fake throws where we know the exact true path and landing spot, so we can test the filter without the camera. It steps the ball forward with gravity and drag until the ball reaches the floor. It then adds stereo noise to each position, using the same noise model as the real camera, and can randomly drop frames to imitate missed detections.

### Baseline Prediction — baseline.py

To check whether the Kalman filter was worth building we made a baseline to compare it against. Our rule was that if the Kalman filter couldn't beat the curve fit, the extra complexity wasn't worth it.

### Evaluation — evaluation.py

The evaluation measures how good the landing predictions are. For each throw we stop watching partway through the flight, make a landing guess with both the Kalman filter and the baseline, and measure the distance on the floor between the guess and the real landing spot. Across all throws we report the median error and how many throws produced no guess at all. Counting throws with no guess matters because a method that only answers on easy throws would look better than it really is. We also check whether the true landing spot fell inside the filter's own 95% ellipse. If the filter is honest about its uncertainty this should happen about 95% of the time.

### Recording Playback — scripts/run_throw.py

The recording playback script works to run our tracking system on a previously recorded throw. We can provide a video file or a folder containing a recorded throw and select the starting and ending frames to focus on the ball’s flight. The script loads the camera calibration and runs the recording through ball detection, triangulation, motion estimation, and landing prediction. It prints the results for each frame, including whether the ball was detected, its world position, whether the measurement was accepted by the Kalman filter, and its predicted landing position. It then creates a new video showing the ball detection and predicted trajectory on the rectified left camera image. Finally, when a prediction is available, it saves a three-dimensional tracking plot and a landing uncertainty plot. This allows us to review the same throw multiple times and compare the tracking results without needing to capture a new recording.

### Visualization and Helper Scripts

The following functions and helper scripts were primarily AI generated through Claude and ChatGPT to help us visualize results, test the system, and prepare recordings for evaluation. Completely implementing each of these ourselves was not in scope for this project.

- tune_detector.py: Displays live camera footage with ball detection overlays and allows us to switch between color and Hough detection to compare their performance. (hand-made).
- verify_calibration.py: Displays live stereo images, ball detections, and three-dimensional positions, with trajectory predictions when a world mapping is available. It also lets us check calibration using a checkerboard. (mostly ai)
- checkerboard_pose.py: Uses the floor checkerboard and selected corners to define the world axes and calculate the camera’s position and orientation. (mostly ai)
- estimate_floor.py: Estimates the floor plane and camera height using three-dimensional positions of a ball rolling on the floor. (some ai assistance)
- record.py: Records stereo camera footage and frame timestamps, saving each throw in its own folder for later testing. (mostly ai)
- grid_ground_truth.py: Uses the floor checkerboard and recorded ball motion to estimate reference landing positions and release and landing frames for evaluation. (some ai assistance)
- simulate_filter.py: Tests the Kalman filter on simulated throws with added measurement noise and checks whether its estimated uncertainty matches its errors (some ai assistance)
- run_evaluation.py: Compares landing prediction errors for the Kalman filter and polynomial baseline using simulated or recorded throws, then saves the results as a plot. (some ai assistance)

## Design Decision

- Decision: How should we predict where the ball will land from noisy stereo measurements?
- Alternatives: Fit curves directly to the raw measurements. Use a Kalman filter with a gravity model to estimate position and velocity, then fly that estimate to the floor with gravity and drag.
- Choice: We built the Kalman filter with a drag based landing predictor and kept the curve fit as a baseline to compare against.
- Reasoning: The filter knows gravity already, so it can make a guess from just 2 measurements, while the curve fit needs at least 3. It also knows which measurements are noisy and can throw out bad detections, and gives an uncertainty ellipse along with its guess. Since a throw only lasts about 7 to 15 frames at 15 fps, being able to predict early mattered a lot.
- Tradeoffs: The filter was much more work to build and tune. What we gained was earlier and more reliable predictions plus an uncertainty estimate not better accuracy late in the flight.
- The ball's mass and drag coefficient are estimates not measurements.
- The filter doesn't beat the curve fit once more than about half the flight is seen.

## Testing and Results

We wrote 21 automated pytest tests covering the Kalman filter, landing predictor, simulation, triangulation, evaluation, and pipeline. All 21 pass. We also ran simulated tests so we ran the filter and baseline on 200 simulated throws with realistic stereo noise and dropped frames.

**Median landing error:**

| Flight seen | Kalman filter | Baseline | Filter: no guess | Baseline: no guess | True landing inside filter's 95% ellipse |
| --- | --- | --- | --- | --- | --- |
| 10% | 119.3 cm |  | 43% | 100% | 90% |
| 25% | 71.1 cm | 70.7 cm | 2% | 24% | 91% |
| 50% | 28.3 cm | 24.8 cm | 0% | 0.5% | 97% |
| 75% | 15.9 cm | 13.9 cm | 0% | 0% | 96% |
| 90% | 11.7 cm | 11.5 cm | 0% | 0% | 97% |

We recorded 20 throws with the ZED 2i over a floor checkerboard and compared the predictions to the checkerboard-based true landing spots.

**Median landing error:**

| Flight seen | Kalman filter | Baseline | Filter: no guess | Baseline: no guess | True landing inside filter's 95% ellipse |
| --- | --- | --- | --- | --- | --- |
| 10% | 49.1 cm |  | 80% | 100% | 75% |
| 25% | 15.6 cm | 21.1 cm | 15% | 70% | 76% |
| 50% | 10.1 cm | 9.2 cm | 0% | 10% | 85% |
| 75% | 8.3 cm | 7.1 cm | 0% | 0% | 95% |
| 90% | 8.3 cm | 7.4 cm | 0% | 0% | 85% |

Early in the flight, the filter makes a guess much more often than the baseline. With 25% of the flight seen on real throws, the filter gave a guess on 85% of throws and the baseline on only 30%, and the filter's guesses were more accurate. By the end of the flight, both methods land within about 7 to 8 cm of the true spot. On clean throws the 3D track plots show the filtered path following the measurements closely.

We did have a few struggles: After about half the flight, the simple curve fit was slightly more accurate than the filter. The filter's 95% ellipse was also too small on real throws, 75% to 95% coverage instead of 95%. In our earlier recordings without the grid, the detector sometimes picked up something other than the ball. The outlier check threw out those points, but the 3D plots for those throws are stretched out by the bad detections. In the 20 grid throws, almost no measurements were thrown out.

The plots below are the outputs of scripts/run_evaluation.py and scripts/run_throw.py. The 3D track, landing ellipse, and overlay for all 20 grid throws are in output/video (N)/ in the repo.

<img width="1600" height="1400" alt="image" src="https://github.com/user-attachments/assets/c5e9feb2-cb46-4eb7-a0aa-f7140d4d0f99" />

*Simulated: landing error and failed predictions vs. how much of the flight was seen (200 throws)*

<img width="1600" height="1400" alt="image" src="https://github.com/user-attachments/assets/ff7c6420-34ae-4c4b-a9be-04c9b37ac36c" />

*Real camera: landing error and failed predictions vs. how much of the flight was seen (20 throws)*
<img width="1800" height="1200" alt="image" src="https://github.com/user-attachments/assets/9355576e-b7f8-4250-b994-c5d4de3ef99b" />

*One real grid throw (video 20). The ball was found in all 11 frames and the final guess landed 2.3 cm from the true spot.*
<img width="1400" height="1200" alt="image" src="https://github.com/user-attachments/assets/75663779-fc4a-4bdf-947f-f4c22db3a1fc" />


*Same throw’s predicted landing point with its 95% uncertainty ellipse*

\*\* look at the repo under output for the gifs of landing prediction as an overlay \*\*

## Challenges

These are some challenges we faced along the way

- The filter locking itself out. Early on, one bad velocity estimate made every later measurement look too surprising, so the filter rejected all of them and never recovered. We fixed this by not running the outlier check for the first 3 updates and by restarting the filter after 5 rejections in a row.
- Predictions landing too far. A gravity only prediction kept overshooting the real landing spot. Our ball is light, so drag is nearly as strong as gravity at 7 m/s. We added quadratic drag to both the simulation and the landing predictor.
- The filter being overconfident. We checked the filter's honesty with NEES on simulated throws. At first, R was calculated at the measured position. When a noisy measurement happened to be too close to the camera, R came out too small, and the filter trusted exactly the measurements it shouldn't have. Calculating R at the predicted position instead brought NEES down from over 6.7 to about 6, which is the target for 6 state values.
- Knowing the true landing spot. At first we estimated the landing spot from the stereo camera itself, which isn't independent of what we were testing. We switched to a floor checkerboard, which gave us both the camera's position and the landing spots in a way that doesn't depend on the stereo depth.

**Still unresolved:**

- On real throws the true landing spot was inside the filter's 95% ellipse only 75% to 95% of the time so the filter is still somewhat overconfident.
- A few throws have NIS above the target of 3 to 4 so the noise settings may still be off and may need to be further tuned

## Future Improvements

If we could improve our project with more time, we would:

- Measure the ball's mass and drag. The drag model drives most of the landing prediction, and right now it relies on estimated values. Dropping the ball from known heights and timing it would give us real numbers.
- Fix the overconfidence on real throws. We'd retune SIGMA_A and the pixel noise on real data instead of only on simulation, so the 95% ellipse actually contains the landing spot about 95% of the time.
- Use a higher frame rate. At 15 fps, a throw only gives 7 to 15 frames. Running at 60 fps would give the filter four times as many measurements even if the video is at lower resolution
- Add drag inside the filter itself as an extended Kalman filter, so its velocity estimate accounts for the ball slowing down between frames instead of treating that as noise.

## Lessons Learned

One of our biggest lessons was to simulate first. Fake throws with known answers made bugs easy to spot, like the filter locking itself out or the noise (R) being set too small. Simulation alone wasn’t enough, though. Real throws were slower, noisier and shorter than we planned for, and because we tuned mostly on simulation, our filter is still overconfident on real data. We also learned that being accurate isn’t enough, the filter has to know how unsure it is, and NEES and NIS let us check that instead of guessing. We first treated drag as optional, but for our light ball it’s almost as strong as gravity, and leaving it out overshot one throw by 58 cm. On the perception side, the ball is only about 10 pixels wide, so detection limits everything else. Hough circles couldn’t find it reliably, and color detection sometimes picked up other red objects. In the future we may attempt to combine both methods for an optimal ball detection.

We also learned that it is important to test against something simple like a baseline. Our curve fit baseline showed what the filter is actually good at, it predicts earlier and gives an uncertainty, but it isn’t more accurate later in the flight. We also learned that ground truth has to be independent. Using the camera to check itself can hide its own errors so we switched to a floor checkerboard.

We also learned that building on good references, like the Kalman and Bayesian Filters in Python book, filterpy, and OpenCV documentation & tutorials can save us from starting from scratch while still allowing us to meet our learning goals.

On our next project, we would set up ground truth before the first recording. We would also check the camera’s frame rate and the ball’s size in pixels before designing around them. We would measure the ball’s drag early instead of using a textbook value, and start tuning on real data sooner. Before recording, we would clear other red objects from the scene and fix the lighting. Finally, we would write up each decision as we make it, not at the end.

## Connection to Real-Life Robotic Systems

Our project relates to real-life robotic systems because it uses camera images to estimate the position and motion of an object and predict where it will go next. A robot could use these outputs to move toward a predicted landing position or prepare to intercept the ball before it arrives. This would require connecting our tracking system to the robot’s motion planning and control system so it can respond to the predictions.

One potential application is basketball, where a robot could predict where a missed shot will land and move into position to collect the rebound. The tracking data could also help players analyze the height and shape of their shots. In tennis or table tennis, a robot could use the ball’s estimated position and velocity to predict where it will cross the racket’s reach and plan a return. In golf, a similar system could track the beginning of a shot and estimate its flight path and landing area, giving players feedback on distance and consistency.

## Contributions:

| Area | Lead | Includes |
| --- | --- | --- |
| Perception | Matthew | Camera capture, calibration, ball detection, stereo triangulation |
| Estimation and prediction | Aditi | Kalman filter, landing-point prediction, error analysis |
| Shared | Both | Deriving and tuning the filter, data collection, visualizations, evaluation, writeup and demo video |

## References

### Kalman filtering and estimation

- R. R. Labbe Jr., Kalman and Bayesian Filters in Python. https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
- R. R. Labbe Jr., FilterPy documentation. https://filterpy.readthedocs.io/
- R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” Journal of Basic Engineering, vol. 82, no. 1, pp. 35–45, 1960.
- Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, Estimation with Applications to Tracking and Navigation. Wiley, 2001. (NEES/NIS consistency tests and chi-square gating)

### Ball detection, camera calibration, triangulation

- OpenCV-Python tutorials and documentation
- https://docs.opencv.org/5.0/py_tutorials/py_tutorials.html

## AI Usage

- Python files themselves indicate where AI assistance was used
- Claude was used for guidance in writing some of the test scripts.
- ChatGPT was used as assistance to write most of the scripts in the scripts folder
- Minimal ai usage in the ball_tracking_machine_vision folder
