"""Kalman filter with a constant-gravity motion model.

State: [x, y, z, vx, vy, vz]
"""

import numpy as np

from ball_tracking_machine_vision.simulation import GRAVITY

# 99% cutoff for a chi-square distribution with 3 degrees of freedom
# A measurement further than this from the prediction is probably a bad detection
GATE_THRESHOLD = 11.34


class BallKalmanFilter:

    def __init__(self, sigma_a, measurement_noise):
        """
        sigma_a: strength of unmodeled acceleration (m/s²), start around 2.
        measurement_noise: 3×3 R matrix (m²), e.g. np.diag(np.array([0.01, 0.01, 0.03])**2)
        """
        self.sigma_a = sigma_a
        self.R = np.asarray(measurement_noise, dtype=float)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])   # we only measure position
        self.x = None
        self.P = None
        self.last_time = None

    def initialize(self, first_measurement, timestamp):
        """Start the filter from the first 3D measurement."""
        self.x = np.concatenate([first_measurement, np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = self.R                # position: as good as one measurement
        self.P[3:, 3:] = 100.0 * np.eye(3)     # velocity: unknown (σ = 10 m/s)
        self.last_time = timestamp

    def predict(self, dt):
        """Propagate the state and covariance forward by dt under gravity."""
        F = transition_matrix(dt)
        self.x = F @ self.x + gravity_input(dt)
        self.P = F @ self.P @ F.T + process_noise(dt, self.sigma_a)

    def update(self, measurement):
        """Correct the state with a new 3D position measurement."""
        y = measurement - self.H @ self.x            # residual --> measured minus predicted
        S = self.H @ self.P @ self.H.T + self.R      # expected size of that residual

        # reject measurements implausibly far from the prediction
        if y @ np.linalg.solve(S, y) > GATE_THRESHOLD:
            return False

        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y

        # Joseph form --> same as (I - KH) P but stays symmetric and positive
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return True

    def get_state(self):
        """Return the current state estimate and covariance."""
        return self.x.copy(), self.P.copy()

    def step(self, measurement, timestamp):
        """Predict to this timestamp, then update. Handles uneven frame gaps."""
        self.predict(timestamp - self.last_time)
        self.last_time = timestamp
        return self.update(measurement)


def transition_matrix(dt):
    """State transition matrix F for time step dt."""
    F = np.eye(6)
    F[:3, 3:] = dt * np.eye(3)   # top-right block --> position += velocity * dt
    return F


def gravity_input(dt):
    """Control term that applies gravity over dt."""
    u = np.zeros(6)
    u[2] = -0.5 * GRAVITY * dt**2   # z position drops by ½ g dt²
    u[5] = -GRAVITY * dt            # z velocity drops by g dt
    return u


def process_noise(dt, sigma_a):
    """Process noise Q -- unmodeled acceleration with standard deviation sigma_a."""
    I = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = dt**4 / 4 * I    # position-position
    Q[:3, 3:] = dt**3 / 2 * I    # position-velocity
    Q[3:, :3] = dt**3 / 2 * I    # velocity-position
    Q[3:, 3:] = dt**2 * I        # velocity-velocity
    return sigma_a**2 * Q
