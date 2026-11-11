import cv2
import time
import numpy as np
import configparser

"""Reading stereo frames from the ZED 2i as a plain USB camera with OpenCV.

The ZED shows up as a normal webcam that returns one side-by-side image
(left | right) per frame.
"""

def open_camera(source, resolution, fps):
    """Open the ZED 2i with cv2.VideoCapture at the given resolution/fps.

    Args:
        source: Camera index or device path.
        resolution: Full frame (width, height), including both stereo views.
        fps: Requested frames per second.
    Returns:
        Open cv2.VideoCapture object.
    """
    camera = cv2.VideoCapture(source,cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Error if video source cant be opened
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open video source: {source}")
    
    width, height = resolution

    # For the ZED, width includes both views side by side.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    
    # Requested settings are not always accepted by the camera.
    # Get the actual width height and fps the camera is using
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Camera settings: {actual_width}x{actual_height}, {actual_fps:g} FPS")

    return camera


def split_side_by_side(frame):
    """Split a side-by-side ZED frame into left and right images.

    Args:
        frame: Image array containing left and right views side by side.
    Returns:
        Tuple (left, right) of image arrays.
    """
    # Frame is height,width,color
    # All frame rows, only half the columns to get left and right frames
    left = frame[:, :frame.shape[1] // 2]
    right = frame[:, frame.shape[1] // 2:]
    return left, right


def read_stereo_frame(camera):
    """Grab one frame and return (left, right, timestamp).

    Args:
        camera: Open cv2.VideoCapture object.
    Returns:
        Tuple (left, right, timestamp), with a monotonic timestamp in seconds.
    """
    ok, frame = camera.read()
    timestamp = time.monotonic()
    # If frame is not read
    if not ok:
        raise RuntimeError("Could not read a camera frame")
    
    left,right = split_side_by_side(frame)
    
    return left, right, timestamp

def load_calibration(conf_path, resolution): 
    """Load the ZED factory calibration file (SN*.conf): intrinsics, baseline.

    Args:
        conf_path: Path to the ZED factory calibration file.
        resolution: Single-view (width, height) in pixels.
    Returns:
        Tuple (K_left, D_left, K_right, D_right, R, T, baseline, resolution).
        K and D are intrinsics and distortion; R and T map left to right.
        T and baseline are in meters.
    """
    # conf_path = "docs/SN35842573.conf"   
    # Set resolution mode 
    # Resolution refers to one view, after splitting the stereo frame.
    modes = {
    (2208, 1242): "2K",
    (1920, 1080): "FHD",
    (1280, 720): "HD",
    (672, 376): "VGA",
    }

    if resolution not in modes:
        raise ValueError(f"Unsupported resolution: {resolution}")

    mode = modes[resolution]
    # Section names for going through the config file
    left_section = f"LEFT_CAM_{mode}"
    right_section = f"RIGHT_CAM_{mode}"
    stereo_section = "STEREO"

    # Open config file
    config = configparser.ConfigParser()
    with open(conf_path) as file:
        config.read_file(file)

    # Function to get the intrinsics
    def get_intrinsics(section):
        """Read intrinsics and distortion for one camera.

        Args:
            section: Camera section name in the calibration file.
        Returns:
            Tuple (K, D): 3x3 intrinsic matrix and distortion coefficients.
        """
        # Grab values from config
        fx = float(config.get(section, "fx")) # horizontal focal length
        fy = float(config.get(section, "fy")) # vertical focal length
        cx = float(config.get(section, "cx")) # horizontal coordinate of optical center
        cy = float(config.get(section, "cy")) # vertical coordinate of optical center'

        # Create intrinsic matrix
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1],])

        # Grab distortion coefficients
        k1 = float(config.get(section, "k1"))
        k2 = float(config.get(section, "k2"))
        p1 = float(config.get(section, "p1", fallback=0.0))
        p2 = float(config.get(section, "p2", fallback=0.0))

        # Create distortion array
        k3 = float(config.get(section, "k3", fallback=0.0))
        D = np.array([k1, k2, p1, p2, k3])

        return K,D
    K_left, D_left = get_intrinsics(left_section)
    K_right, D_right = get_intrinsics(right_section)


    # Read the calibration describing the relationship between the cameras.
    stereo = config[stereo_section]

    # Horizontal separation of the cameras in meters.
    baseline = stereo.getfloat("Baseline") / 1000.0

    # Translation used to transform left-camera coordinates into right-camera
    # coordinates. 
    T = np.array([
        -baseline,
        stereo.getfloat("TY") / 1000.0,
        stereo.getfloat("TZ") / 1000.0,
    ], dtype=np.float64).reshape(3, 1)

    # The cameras are almost parallel, but have a small relative rotation.
    # These three file values form a rotation vector.
    rotation_vector = np.array([
        stereo.getfloat(f"RX_{mode}"),
        stereo.getfloat(f"CV_{mode}"),
        stereo.getfloat(f"RZ_{mode}"),
    ], dtype=np.float64)

    # Convert the rotation vector to the matrix OpenCV expects.
    R, _ = cv2.Rodrigues(rotation_vector)

    # Return everything so later functions can use it.
    return K_left, D_left, K_right, D_right, R, T, baseline, resolution
    
    

def build_rectification_maps(image_size):
    """Precompute the OpenCV remap tables for rectifying left and right.

    Args:
        image_size: Single-view (width, height) in pixels.
    Returns:
        Tuple (left_map_x, left_map_y, right_map_x, right_map_y, P1, P2, Q).
        Includes remap arrays, projection matrices, and disparity-to-3D matrix.
    """
    # Load calibration data
    K_left, D_left, K_right, D_right, R, T, baseline, resolution = load_calibration("docs/SN35842573.conf",image_size)
    # Get the rectification transforms for each eye of stereo camera
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_left, D_left, K_right, D_right, image_size, R, T)

    # cv2.initUndistoryRectifyMap Computes the undistortion and rectification transformation map.
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)

    right_map_x, right_map_y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)

    return left_map_x, left_map_y, right_map_x, right_map_y, P1, P2, Q

def rectify(left, right, rectification_maps):
    """Rectify a stereo pair so epipolar lines are horizontal.
    Args:
        left: The image of the left camera
        right: The image of the right camera:
        rectification_maps: The rectification map for each of the left and right camera
    Returns:
        Tuple (left_remap, right_remap) of rectified image arrays.
    """
    left_map_x, left_map_y, right_map_x, right_map_y,P1,P2,Q = rectification_maps
    # Remap so the ball in both images are at the same coordinate
    left_remap = cv2.remap(left,left_map_x,left_map_y,cv2.INTER_LINEAR)
    right_remap = cv2.remap(right,right_map_x,right_map_y,cv2.INTER_LINEAR)
    return left_remap, right_remap
    


# This function was developed primarily using ChatGPT. It serves as a test to check
# our implementation of the previous functions worked properly.   
def verify_calibration_checkerboard(
    left, right, rectification_maps, square_size, pattern_size
):
    """Check stereo calibration against a checkerboard.

    Args:
        left, right: Raw stereo images, before rectification.
        rectification_maps: Tuple from build_rectification_maps().
        square_size: Physical square width in meters, e.g. 0.025.
        pattern_size: Internal corners as (columns, rows), e.g. (7, 5).

    Returns:
        Dictionary containing measurements and annotated images.
        A successful measurement does not automatically mean calibration passes.
    """
    columns, rows = pattern_size
    if columns < 3 or rows < 3:
        raise ValueError("Use at least 3 internal corners in each direction.")
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be a positive number in meters.")

    left_map_x, left_map_y, right_map_x, right_map_y, P1, P2, Q = (
        rectification_maps
    )

    # Images must have the same dimensions used when building the maps.
    if (left.shape[:2] != left_map_x.shape
            or right.shape[:2] != right_map_x.shape):
        raise ValueError("Image dimensions do not match the rectification maps.")

    left_rect, right_rect = rectify(left, right, rectification_maps)
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found_left, corners_left = cv2.findChessboardCorners(
        left_gray, pattern_size, flags
    )
    found_right, corners_right = cv2.findChessboardCorners(
        right_gray, pattern_size, flags
    )

    report = {
        "success": False,
        "found_left": bool(found_left),
        "found_right": bool(found_right),
        "left_preview": left_rect.copy(),
        "right_preview": right_rect.copy(),
    }

    if not (found_left and found_right):
        report["message"] = "The complete checkerboard was not found in both views."
        return report

    # Refine corner positions to fractions of a pixel.
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners_left = cv2.cornerSubPix(
        left_gray, corners_left, (5, 5), (-1, -1), criteria
    )
    corners_right = cv2.cornerSubPix(
        right_gray, corners_right, (5, 5), (-1, -1), criteria
    )

    # A checkerboard detector can enumerate the board in opposite directions.
    # Try the original and reversed right-corner order, choosing the one
    # with the smaller vertical mismatch in these rectified views.
    reverse_right = corners_right[::-1].copy()
    error_original = np.mean(
        np.abs(corners_left[:, 0, 1] - corners_right[:, 0, 1])
    )
    error_reversed = np.mean(
        np.abs(corners_left[:, 0, 1] - reverse_right[:, 0, 1])
    )
    if error_reversed < error_original:
        corners_right = reverse_right

    # Draw ordered corners so correspondence can also be checked visually.
    cv2.drawChessboardCorners(
        report["left_preview"], pattern_size, corners_left, True
    )
    cv2.drawChessboardCorners(
        report["right_preview"], pattern_size, corners_right, True
    )

    vertical_errors = np.abs(
        corners_left[:, 0, 1] - corners_right[:, 0, 1]
    )

    # triangulatePoints expects a 2×N array of pixel coordinates per camera.
    points_left = corners_left.reshape(-1, 2).T.astype(np.float64)
    points_right = corners_right.reshape(-1, 2).T.astype(np.float64)

    homogeneous = cv2.triangulatePoints(P1, P2, points_left, points_right)

    # Convert homogeneous (X, Y, Z, W) coordinates into ordinary 3D points.
    w = homogeneous[3]
    if (not np.all(np.isfinite(homogeneous))
            or np.any(np.abs(w) < 1e-12)):
        report["message"] = "Some corners could not be triangulated reliably."
        return report

    points_3d = (homogeneous[:3] / w).T

    # Points should be in front of both rectified cameras.
    points_h = np.column_stack((points_3d, np.ones(len(points_3d))))
    if (np.any((P1 @ points_h.T)[2] <= 0)
            or np.any((P2 @ points_h.T)[2] <= 0)):
        report["message"] = (
            "Some corners reconstructed behind a camera. "
            "Check corner correspondence and stereo calibration."
        )
        return report

    # Arrange the points into the checkerboard grid and measure neighboring
    # corners horizontally and vertically. Both should be square_size apart.
    grid = points_3d.reshape(rows, columns, 3)
    horizontal = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2)
    vertical = np.linalg.norm(grid[1:] - grid[:-1], axis=2)
    spacings = np.concatenate((horizontal.ravel(), vertical.ravel()))

    absolute_errors = np.abs(spacings - square_size)

    report.update({
        "success": True,
        "message": "Measurements computed; inspect errors and corner ordering.",
        "mean_vertical_error_px": float(vertical_errors.mean()),
        "max_vertical_error_px": float(vertical_errors.max()),
        "expected_spacing_m": float(square_size),
        "mean_spacing_m": float(spacings.mean()),
        "mean_absolute_spacing_error_m": float(absolute_errors.mean()),
        "mean_absolute_spacing_error_percent": float(
            100 * absolute_errors.mean() / square_size
        ),
        "points_3d": points_3d,
    })
    return report
    
