"""Turning a pair of 2D detections into a 3D ball position."""


def compute_disparity(left_point, right_point):
    """Horizontal pixel offset between left and right detections."""
    pass


def triangulate(left_point, right_point, calibration):
    """Return the ball's 3D position (x, y, z) in the camera frame."""
    pass


def camera_to_world(point, camera_pose):
    """Transform a camera-frame point into the floor/world frame."""
    pass
