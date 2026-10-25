"""Saving and loading recorded throws (replaces ROS bags).

Each throw is a folder with the raw side-by-side video and a CSV of
per-frame timestamps, so the whole pipeline can be rerun offline.
"""


def record_throw(camera, output_dir, duration):
    """Record one throw to output_dir/video.avi and output_dir/timestamps.csv."""
    pass


def load_throw(throw_dir):
    """Yield (left, right, timestamp) for every frame of a recorded throw."""
    pass


def list_throws(data_dir):
    """Return every recorded throw folder in data_dir."""
    pass
