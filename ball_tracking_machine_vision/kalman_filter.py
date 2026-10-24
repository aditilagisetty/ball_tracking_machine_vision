"""Kalman filter with a constant-gravity motion model.

State: [x, y, z, vx, vy, vz]
"""


class BallKalmanFilter:

    def __init__(self, process_noise, measurement_noise):
        pass

    def initialize(self, first_measurement, timestamp):
        """Start the filter from the first 3D measurement."""
        pass

    def predict(self, dt):
        """Propagate the state and covariance forward by dt under gravity."""
        pass

    def update(self, measurement):
        """Correct the state with a new 3D position measurement."""
        pass

    def get_state(self):
        """Return the current state estimate and covariance."""
        pass


def transition_matrix(dt):
    """State transition matrix F for time step dt."""
    pass


def gravity_input(dt):
    """Control term that applies gravity over dt."""
    pass
