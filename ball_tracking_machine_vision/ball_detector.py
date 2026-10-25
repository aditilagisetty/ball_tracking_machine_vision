"""Finding the tennis ball in a single image.

Two approaches to compare especially on motion blurred frames
color thresholding and the Hough circle transform.
"""

# I suggest making the color mask first (honestly work on functions in the order they are in)
# you might have to use tune_detector.py too


def make_color_mask(image, hsv_low, hsv_high): 
    """Threshold the image in HSV to isolate ball-colored pixels."""
    pass


def detect_ball_color(image, hsv_low, hsv_high):
    """Detect the ball as the largest ball-colored blob. Returns (u, v, radius) or None."""
    pass


def detect_ball_hough(image, hough_params):
    """Detect the ball with cv2.HoughCircles. Returns (u, v, radius) or None."""
    pass


def detect_ball(image, method):
    """Run the chosen detection method on one image."""
    pass


def detect_stereo(left, right, method):
    """Detect the ball in both images of a rectified stereo pair."""
    pass
