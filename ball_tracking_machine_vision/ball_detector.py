"""Finding the basketball in a single image.

Two approaches to compare especially on motion blurred frames
color thresholding and the Hough circle transform.
"""
import cv2
import numpy as np


# I suggest making the color mask first (honestly work on functions in the order they are in)
# you might have to use tune_detector.py too


def make_color_mask(image, hsv_low, hsv_high): 
    """Threshold the image in HSV to isolate ball-colored pixels.
    Args:
        image: String containing image path
        hsv_low: Low of color range to detect
        hsv_high: High of color range to detect
    Returns: 
        Color mask array 
    """
    # Convert image to HSV
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # Create color mask
        # cv2.inRange checks if pixel values of an input image fall between a specified lower and upper boundary
        # Then returns binary mask where matching pixels are set to 255 (White) and non matching 0 (Black)
    return cv2.inRange(
        hsv,
        np.array(hsv_low, dtype=np.uint8),
        np.array(hsv_high, dtype=np.uint8),
    )

def detect_ball_color(image):
    """Detect the ball as the largest ball-colored blob. Returns (u, v, radius) or None."""
    # Create color mask
    mask1 = make_color_mask(image, RED_LOW_1, RED_HIGH_1)
    mask2 = make_color_mask(image, RED_LOW_2, RED_HIGH_2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Find the outer boundaries of white blobs in the binary mask (contours)
        # cv2.findContours extracts continuous boundary lines sharing the same color/intensity from binary imgs.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
    # If no contours, return none
    if not contours:
        return None

    # Find the contour with the largest enclosed area
        # cv2.contourArea calculates the area of a countour
    largest = max(contours, key=cv2.contourArea)
    # Get the center coordinates and radius of the circle
        # cv2.minEnclosingCircle finds the smallest possible circle that covers a contour.
    (u, v), radius = cv2.minEnclosingCircle(largest)
    return u,v ,radius



def detect_ball_hough(image, hough_params):
    """Detect the ball with cv2.HoughCircles. Returns (u, v, radius) or None."""\
    # Convert image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Then blur the image to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 2)
    cv2.imshow("Hough input", blurred_img)

        # cv2.HoughCircles 
    circles = cv2.HoughCircles(
        blurred_img,
        cv2.HOUGH_GRADIENT,
        **hough_params, )

    if circles is None:
        return None

    # Initially choose the largest detected circle.
    largest = max(circles[0], key=lambda circle: circle[2])
    u, v, radius = largest


    return float(u), float(v), float(radius)

    
# Orange basketball HSV starting values; tune for lighting.
# Keep the existing names for compatibility with tune_detector.py.
# Both ranges cover orange so the second mask does not also select red.
# Basketball HSV detection
RED_LOW_1 = (0, 190, 160)
RED_HIGH_1 = (10, 255, 255)

RED_LOW_2 = (0, 190, 160)
RED_HIGH_2 = (10, 255, 255)

HOUGH_PARAMS = {
    "dp": 1.2,
    "minDist": 30,
    "param1": 150,
    "param2": 25,
    "minRadius": 15,
    "maxRadius": 150,
}
def detect_ball(image, method):
    """Run the chosen detection method on one image."""
    if method == "color":
        return detect_ball_color(image)
    if method == "hough":
        return detect_ball_hough(image, HOUGH_PARAMS)
    raise ValueError(f"Unknown detection method: {method}")


def detect_stereo(left, right, method):
    """Detect the ball in both images of a rectified stereo pair.
    Args:
        left: left image from stereo camera
        right: right image from stereo camera
        method: the detection method (either color or hough)
    Returns:
        (left_detection, right_detection), each containing (u, v, radius).
    """
    # Detect ball for each image
    left_detection = detect_ball(left, method)
    right_detection = detect_ball(right, method)

    # Return none if no detection
    if left_detection is None or right_detection is None:
        return None, None

    return left_detection, right_detection
