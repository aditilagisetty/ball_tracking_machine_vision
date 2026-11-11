"""Triangulation tests with known synthetic geometry."""

import numpy as np

from ball_tracking_machine_vision.triangulation import compute_disparity, triangulate


# Rectified cameras with an 800-pixel focal length and 0.12-meter baseline.
P1 = np.array([[800, 0, 320, 0], [0, 800, 240, 0], [0, 0, 1, 0]], dtype=float)
P2 = np.array([[800, 0, 320, -96], [0, 800, 240, 0], [0, 0, 1, 0]], dtype=float)


def test_point_on_optical_axis():
    """A point straight ahead should triangulate to (0, 0, z)."""
    # A point 2 meters ahead projects to the left optical center.
    left = (320, 240, 10)
    right = (272, 240, 10)
    point = triangulate(left, right, P1, P2)
    np.testing.assert_allclose(point, (0, 0, 2), atol=1e-9)


def test_depth_from_disparity():
    """Depth should match baseline * focal / disparity."""
    # Check several depths using detections away from the optical center.
    for disparity in (16, 32, 64):
        left = (400, 260, 10)
        right = (400 - disparity, 260, 10)
        measured_disparity = compute_disparity(left, right)
        assert measured_disparity == disparity
        expected_depth = 0.12 * 800 / disparity
        point = triangulate(left, right, P1, P2)
        assert point is not None
        np.testing.assert_allclose(point[2], expected_depth, atol=1e-9)
