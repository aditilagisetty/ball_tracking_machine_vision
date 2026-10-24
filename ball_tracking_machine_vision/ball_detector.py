"""Finding the tennis ball in a single image."""


def make_color_mask(image, hsv_low, hsv_high):
    """Threshold the image in HSV to isolate ball-colored pixels."""
    pass


def find_ball(mask):
    """Return the ball centre (u, v) and radius in pixels, or None."""
    pass


def detect_ball(image):
    """Full detection on one image: mask, then find the ball."""
    pass


def detect_stereo(left, right):
    """Detect the ball in both images of a stereo pair."""
    pass
