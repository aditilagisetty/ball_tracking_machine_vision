"""Reading stereo frames from the ZED 2i as a plain USB camera with OpenCV.

The ZED shows up as a normal webcam that returns one side-by-side image
(left | right) per frame.
"""


def open_camera(device_id, resolution, fps):
    """Open the ZED 2i with cv2.VideoCapture at the given resolution/fps."""
    pass


def read_stereo_frame(camera):
    """Grab one frame and return (left, right, timestamp)."""
    pass


def split_side_by_side(frame):
    """Split a side-by-side ZED frame into left and right images."""
    pass


def load_calibration(conf_path, resolution):
    """Load the ZED factory calibration file (SN*.conf): intrinsics, baseline."""
    pass


def build_rectification_maps(calibration, image_size):
    """Precompute the OpenCV remap tables for rectifying left and right."""
    pass


def rectify(left, right, rectification_maps):
    """Rectify a stereo pair so epipolar lines are horizontal."""
    pass


def verify_calibration_checkerboard(left, right, calibration, square_size):
    """Triangulate checkerboard corners and compare against known spacing."""
    pass
