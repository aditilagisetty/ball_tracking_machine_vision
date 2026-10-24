"""Landing-point error vs. fraction of flight observed."""


def load_throw(bag_path):
    """Load one recorded throw from a ROS 2 bag."""
    pass


def load_ground_truth(path):
    """Load the measured true landing points (10 cm grid + phone video)."""
    pass


def run_throw(throw):
    """Run detection, triangulation, and filtering on one throw."""
    pass


def landing_error(predicted, true_landing):
    """Distance between predicted and true landing point."""
    pass


def error_vs_fraction_observed(results, ground_truth):
    """Landing error at each fraction of the flight, for filter and baseline."""
    pass


def evaluate_all(throw_paths, ground_truth_path):
    """Run the full evaluation over all recorded throws."""
    pass
