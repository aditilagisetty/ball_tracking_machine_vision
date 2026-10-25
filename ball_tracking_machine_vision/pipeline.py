"""Ties detection, triangulation, filtering, and prediction together for one throw."""


def track_throw(frames, calibration, floor_height):
    """Run the full pipeline on a throw.

    Returns per-frame measurements, filter estimates, and landing predictions.
    """
    pass


def process_frame(left, right, timestamp, kalman_filter, calibration):
    """Detect, triangulate, and filter a single stereo frame."""
    pass
