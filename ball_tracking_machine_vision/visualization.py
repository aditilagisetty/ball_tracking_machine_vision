"""Plots and overlays of what the system sees and predicts."""
import cv2
import numpy as np


def draw_detection(image, detection):
   """Draw the detected ball on a camera frame.
   Args:
       image: BGR image array.
       detection: Center (u, v) and radius in pixels, or None.
   Returns:
       Image copy with the ball outline and center, if detected.
   """
   display = image.copy()
   # if no detection, just return the displayed frame
   if detection is None:
       return display
   
   # Get radius and center values from detection
   u,v,radius = detection
   center = (round(u),round(v))

   # Create a green outline and red center dot on the ball
   cv2.circle(display,center,round(radius),(0,255,0),2)
   cv2.circle(display,center,3,(0,0,255),-1)
  
   return display
  
# Some AI assistance used to make this function 
def draw_predicted_path(image, trajectory, P1, camera_pose):
   """Project the predicted 3D path back onto the image.
   Args:
     image: the photo frame
     trajectory: Nx3 array of world positions in meters.
     P1: Projection matrix from build_rectification_maps().
     camera_pose: (R_camera_to_world, camera_position_world).
   Returns:
     display: The new frame of the image with path prediction drawn
   """
   
   display = image.copy()

 # if no detection, just return the displayed frame
   if trajectory is None:
       return display

    # World coords from trajectory
   world_points = np.asarray(trajectory, dtype=np.float64)
   if world_points.size == 0:
       return display
   if world_points.ndim != 2 or world_points.shape[1] != 3:
       raise ValueError("trajectory must have shape (N, 3)")

    # Get the rotation and translation from camera pose and reshape
   rotation, translation = camera_pose
   rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
   translation = np.asarray(translation, dtype=np.float64).reshape(3)

   # Reverse camera_to_world(): subtract the camera position, then
   # apply the inverse rotation. A rotation's inverse is its transpose.
   camera_points = (rotation.T @ (world_points - translation).T).T

   # Append 1 to each point so it can be multiplied by the 3×4 P1.
   points_h = np.column_stack((
       camera_points,
       np.ones(len(camera_points)),
   ))
   projected = (P1 @ points_h.T).T

   # Ignore points behind the camera or containing invalid values.
   valid = (
       np.all(np.isfinite(projected), axis=1)
       & (projected[:, 2] > 1e-6)
   )

   pixels = np.full((len(projected), 2), np.nan)
   pixels[valid] = projected[valid, :2] / projected[valid, 2, None]

   height, width = display.shape[:2]

   # Connect consecutive predictions. Skip invalid points rather than
   # drawing across gaps, and clip lines at the image boundaries.
   for index in range(1, len(pixels)):
       if not (valid[index - 1] and valid[index]):
           continue

       segment = pixels[index - 1:index + 1]
       if not np.all(np.isfinite(segment)):
           continue


       # Avoid overflowing OpenCV's integer drawing coordinates.
       if np.any(np.abs(segment) > 1e9):
           continue

       start = tuple(int(round(value)) for value in segment[0])
       end = tuple(int(round(value)) for value in segment[1])


       visible, start, end = cv2.clipLine(
           (0, 0, width, height), start, end
       )
       if visible:
           cv2.line(display, start, end, (0, 255, 255), 2, cv2.LINE_AA)


   return display
  
# AI assistance used to develop the functions below

def _positions(values, name):
   """Normalize pipeline positions/states, preserving missing frames as gaps.

   Args:
       values: Positions, six-element states, a position/state dictionary, or None.
       name: Label used in error messages.
   Returns:
       Nx3 position array, with NaN rows for missing frames.
   """
   if values is None:
       return np.empty((0, 3))
   if isinstance(values, dict):
       if 'position' in values:
           values = values['position']
       elif 'state' in values:
           values = values['state']
       else:
           raise ValueError(f"{name} needs a 'position' or 'state' array")
   positions = []
   for value in values:
       if value is None:
           positions.append([np.nan] * 3)
           continue
       row = np.asarray(value, dtype=float)
       if row.shape not in ((3,), (6,)):
           raise ValueError(f"{name} entries must be XYZ positions or six-element states")
       positions.append(row[:3])
   return np.asarray(positions, dtype=float).reshape(-1, 3)




def plot_track_3d(measurements, estimates, prediction):
   """Return (figure, axes) comparing world-frame tracks in meters.

   Args:
       measurements: Nx3 positions or the pipeline's measurements dictionary.
       estimates: Nx3 positions or Nx6 filter states; None entries mark gaps.
       prediction: Nx3 path, propagate_trajectory() dictionary, predict_landing()
           dictionary (draws its landing point), or None.
   Returns:
       Tuple (figure, axes) containing the 3D track plot.

   Use plt.show() to display, or figure.savefig(...) to save the result.
   """
   # Import plotting only when needed so live OpenCV overlays stay lightweight.
   import matplotlib.pyplot as plt


   fig = plt.figure(figsize=(9, 6))
   ax = fig.add_subplot(111, projection='3d')
   measured = _positions(measurements, 'measurements')
   filtered = _positions(estimates, 'estimates')
   landing = None
   if isinstance(prediction, dict) and 'point' in prediction:
       landing = np.asarray(prediction['point'], dtype=float).reshape(3)
       predicted = np.empty((0, 3))
   else:
       predicted = _positions(prediction, 'prediction')


   all_points = []
   for points, label, color, style in (
       (measured, 'Measured', 'tab:blue', 'points'),
       (filtered, 'Filtered', 'tab:orange', '-'),
       (predicted, 'Predicted path', 'tab:green', '--'),
   ):
       valid = np.all(np.isfinite(points), axis=1)
       if not valid.any():
           continue
       all_points.append(points[valid])
       if style == 'points':
           ax.scatter(*points[valid].T, s=18, alpha=0.6, color=color, label=label)
       else:
           # NaNs break the line instead of connecting across missing estimates.
           clean = points.copy()
           clean[~valid] = np.nan
           ax.plot(*clean.T, style, color=color, label=label)
   if landing is not None and np.all(np.isfinite(landing)):
       ax.scatter(*landing, marker='x', s=90, color='tab:red', label='Predicted landing')
       all_points.append(landing[None, :])


   # Equal physical scale prevents a short vertical motion looking enormous.
   if all_points:
       points = np.vstack(all_points)
       low, high = points.min(axis=0), points.max(axis=0)
       center = (low + high) / 2
       half_span = max(float(np.max(high - low)) * 0.55, 0.05)
       ax.set_xlim(center[0] - half_span, center[0] + half_span)
       ax.set_ylim(center[1] - half_span, center[1] + half_span)
       ax.set_zlim(center[2] - half_span, center[2] + half_span)
       ax.legend()
   else:
       ax.text2D(0.5, 0.5, 'No valid track data', transform=ax.transAxes, ha='center')
   ax.set_box_aspect((1, 1, 1))
   ax.set(xlabel='World x (m)', ylabel='World y (m)', zlabel='World z (m)',
          title='Ball trajectory')
   fig.tight_layout()
   return fig, ax




def plot_landing_uncertainty(landing_point, covariance):
   """Return (figure, axes) with an approximate 95% Gaussian landing ellipse.

   Args:
       landing_point: World (x, y) or (x, y, z), in meters.
       covariance: 2x2 covariance of landing x/y, in square meters, as returned
           by predict_landing(). This is NOT the full filter-state covariance.
   Returns:
       Tuple (figure, axes) containing the landing uncertainty plot.
   """
   import matplotlib.pyplot as plt
   from matplotlib.patches import Ellipse


   point = np.asarray(landing_point, dtype=float)
   cov = np.asarray(covariance, dtype=float)
   if point.shape not in ((2,), (3,)) or not np.all(np.isfinite(point)):
       raise ValueError('landing_point must be a finite 2D or 3D position')
   if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
       raise ValueError('covariance must be a finite 2x2 matrix')
   if not np.allclose(cov, cov.T, atol=1e-10):
       raise ValueError('covariance must be symmetric')


   # Eigenvectors give ellipse directions; eigenvalues give variance along them.
   eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2)
   if eigenvalues.min() < -1e-10:
       raise ValueError('covariance must be positive semidefinite')
   order = np.argsort(eigenvalues)[::-1]
   eigenvalues = np.maximum(eigenvalues[order], 0)
   eigenvectors = eigenvectors[:, order]
   # 5.991 is the 95% chi-square threshold for TWO dimensions, not 1.96².
   radii = np.sqrt(5.991464547 * eigenvalues)
   angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))


   fig, ax = plt.subplots(figsize=(7, 6))
   ellipse = Ellipse(point[:2], width=2 * radii[0], height=2 * radii[1],
                     angle=angle, facecolor='tab:blue', edgecolor='tab:blue',
                     alpha=0.25, label='Approximate 95% ellipse')
   ax.add_patch(ellipse)
   ax.scatter(*point[:2], marker='x', color='tab:red', s=80, label='Predicted landing')
   margin = max(float(radii.max()) * 1.2, 0.05)
   ax.set_xlim(point[0] - margin, point[0] + margin)
   ax.set_ylim(point[1] - margin, point[1] + margin)
   ax.set_aspect('equal', adjustable='box')
   ax.set(xlabel='World x (m)', ylabel='World y (m)', title='Landing uncertainty')
   ax.grid(alpha=0.3)
   ax.legend()
   fig.tight_layout()
   return fig, ax




def plot_error_vs_fraction(error_table):
   """Return (figure, axes) for evaluation.error_vs_fraction_observed() rows.

   Uses fraction, filter_median, baseline_median; errors are supplied in meters.
   None errors remain gaps, never zero. If failure counts are present, a second
   panel shows how many throws produced no prediction, preventing survivor bias.

   Args:
       error_table: Evaluation rows with observed fractions and median errors.
   Returns:
       Tuple (figure, axes), where axes is the landing-error panel.
   """
   import matplotlib.pyplot as plt


   rows = sorted(list(error_table), key=lambda row: row['fraction'])
   fractions = np.asarray([row['fraction'] for row in rows], dtype=float)
   if not np.all(np.isfinite(fractions)) or np.any((fractions < 0) | (fractions > 1)):
       raise ValueError('Observed fractions must lie between 0 and 1')
   with_failures = bool(rows) and all(
       all(key in row for key in ('filter_failed', 'baseline_failed', 'n_throws'))
       for row in rows
   )
   fig, axes = plt.subplots(2 if with_failures else 1, 1,
                            figsize=(8, 7 if with_failures else 5), squeeze=False,
                            sharex=True)
   ax = axes[0, 0]
   for prefix, label, color in (
       ('filter', 'Kalman filter', 'tab:blue'),
       ('baseline', 'Polynomial baseline', 'tab:orange'),
   ):
       values = np.asarray([
           np.nan if row[prefix + '_median'] is None else row[prefix + '_median']
           for row in rows
       ], dtype=float)
       if np.any(values[np.isfinite(values)] < 0):
           raise ValueError('Landing errors cannot be negative')
       ax.plot(fractions * 100, values * 100, 'o-', color=color, label=label)
       if with_failures:
           rates = [100 * row[prefix + '_failed'] / row['n_throws']
                    if row['n_throws'] > 0 else np.nan for row in rows]
           axes[1, 0].plot(fractions * 100, rates, 'o-', color=color, label=label)
   ax.set(ylabel='Median landing error (cm)', title='Landing prediction accuracy')
   ax.set_ylim(bottom=0)
   ax.grid(alpha=0.3)
   ax.legend()
   if not rows:
       ax.text(0.5, 0.5, 'No evaluation data', transform=ax.transAxes, ha='center')
   if with_failures:
       axes[1, 0].set(ylabel='Throws without a prediction (%)', ylim=(-2, 102))
       axes[1, 0].grid(alpha=0.3)
   axes[-1, 0].set(xlabel='Flight observed (%)', xlim=(0, 100))
   fig.tight_layout()
   return fig, ax



