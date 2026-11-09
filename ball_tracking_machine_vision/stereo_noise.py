"""How noisy a stereo 3D measurement is, from the camera's geometry.

Stereo depth comes from disparity: how far the ball shifts between the left
and right images. Depth is baseline * focal / disparity, so a fixed error of
a pixel or so in disparity turns into a depth error that grows with distance
squared. Sideways error only grows linearly with distance. So a stereo camera
is much better at "where across the image" than "how far away":

    sideways noise  σ_side  ≈ Z  · σ_px / f
    depth noise     σ_depth ≈ Z² · σ_px / (f · b)

The simulation uses this to add realistic noise, and the filter uses it to
know how much to trust each measurement (its R matrix). The real pipeline
will use the same functions with the real camera's numbers.

Frames:
    camera frame: OpenCV convention, x right, y down, z forward (depth).
    world frame:  z up, origin on the floor (same as the rest of the project).
"""

import numpy as np

# ZED 2i numbers
# baseline and focal length are from the calibration file
# the rest are PLACEHOLDERS until we measure them

# distance between the left and right lenses (m) 
BASELINE = 0.1198

FOCAL_LENGTH_PX = 265.9

# How far off the ball detector is in pixels
PIXEL_NOISE = 2.5

# Where the simulated camera is in the world frame and what it's pointed at (m)
CAMERA_POSITION = np.array([1.0, -1.5, 0.77])
CAMERA_TARGET = np.array([1.0, 0.0, 0.77])


def look_at(camera_position, target):
    """Rotation matrix from camera frame to world frame for a camera aimed at target.

    Each column is one camera axis written in world coordinates:
        column 0: camera x (right)
        column 1: camera y (down)
        column 2: camera z (forward, the depth direction)
    So rotation @ v_camera = v_world, and rotation.T @ v_world = v_camera.
    Assumes the camera is level (not rolled) and not pointing straight up or down.
    """
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)     # perpendicular to both forward and up
    right = right / np.linalg.norm(right)

    down = np.cross(forward, right)         # completes the right-handed frame

    return np.column_stack([right, down, forward])


def make_camera(position=CAMERA_POSITION, target=CAMERA_TARGET,
                focal_length_px=FOCAL_LENGTH_PX, baseline=BASELINE,
                pixel_noise=PIXEL_NOISE):
    """Bundle everything needed to compute stereo noise into one dict.

    Defaults are the placeholders above; pass real values to override.
    """
    return {
        'position': np.asarray(position, dtype=float),
        'rotation': look_at(np.asarray(position, dtype=float), np.asarray(target, dtype=float)),
        'focal_length_px': focal_length_px,
        'baseline': baseline,
        'pixel_noise': pixel_noise,
    }


def depth_of(point, camera):
    """Distance of a world point along the camera's forward axis (Z in the formulas)."""
    point_in_camera = camera['rotation'].T @ (point - camera['position'])
    return point_in_camera[2]


def stereo_noise_std(depth, camera):
    """Standard deviation of a stereo measurement in the camera frame, (σx, σy, σz).

    x and y (across the image) share the sideways formula; z is depth.
    """
    f = camera['focal_length_px']
    b = camera['baseline']
    s = camera['pixel_noise']

    sigma_side = depth * s / f              # grows linearly with distance
    sigma_depth = depth**2 * s / (f * b)    # grows with distance squared
    return np.array([sigma_side, sigma_side, sigma_depth])


def measurement_covariance(point, camera):
    """3×3 measurement noise covariance R for a ball at this world point.

    The noise is simple in the camera frame (a diagonal matrix: independent
    noise along right, down, forward). The filter works in the world frame,
    so rotate it: R_world = rotation · R_camera · rotationᵀ. If the camera
    looks along a world axis, R_world is still diagonal; if it's tilted, the
    big depth noise gets shared between world axes (off-diagonal terms).
    """
    sigma = stereo_noise_std(depth_of(point, camera), camera)
    R_camera = np.diag(sigma**2)
    rotation = camera['rotation']
    return rotation @ R_camera @ rotation.T
