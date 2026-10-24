"""Reading stereo frames from the ZED 2i (as a plain USB camera) or a bag."""


def open_camera(device_id):
    """Open the ZED 2i as a USB camera."""
    pass


def read_stereo_frame(camera):
    """Grab one side-by-side frame and return (left, right, timestamp)."""
    pass


def split_side_by_side(frame):
    """Split a side-by-side ZED frame into left and right images."""
    pass


def load_calibration(path):
    """Load the ZED factory calibration (intrinsics, baseline)."""
    pass


def rectify(left, right, calibration):
    """Rectify a stereo pair so epipolar lines are horizontal."""
    pass
