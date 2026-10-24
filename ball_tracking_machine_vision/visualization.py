"""Plots and overlays of what the system sees and predicts."""


def draw_detection(image, detection):
    """Draw the detected ball on a camera frame."""
    pass


def draw_predicted_path(image, trajectory, calibration):
    """Project the predicted 3D path back onto the image."""
    pass


def plot_track_3d(measurements, estimates, prediction):
    """3D plot of measured track, filtered track, and predicted path."""
    pass


def plot_landing_uncertainty(landing_point, covariance):
    """Draw the landing-point uncertainty ellipse on the floor plane."""
    pass


def plot_error_vs_fraction(error_table):
    """The main MVP plot: landing error vs. fraction of flight observed."""
    pass
