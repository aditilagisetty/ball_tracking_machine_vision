"""Kalman filter with a constant-gravity motion model.

State: [x, y, z, vx, vy, vz]
"""

import numpy as np

from ball_tracking_machine_vision.simulation import GRAVITY

# 99% cutoff for a chi-square distribution with 3 degrees of freedom
# A measurement further than this from the prediction is probably a bad detection
GATE_THRESHOLD = 11.34

# Don't gate the first few updates. With only 2-3 points the velocity estimate
# can be badly off, and gating on a bad prediction rejects good measurements.
GATE_WARMUP_UPDATES = 10

# If this many measurements in a row get rejected, assume the filter has gone
# wrong (not the camera) and restart it from the latest measurement.
# Otherwise one bad velocity estimate can lock the filter out for the rest of the throw.
MAX_CONSECUTIVE_REJECTIONS = 5


class BallKalmanFilter:

    def __init__(self, sigma_a, measurement_noise):
        """
        sigma_a: strength of unmodeled acceleration (m/s²), start around 2.
        measurement_noise: 3×3 R matrix (m²), e.g. np.diag(np.array([0.01, 0.01, 0.03])**2).
            Used whenever a measurement doesn't come with its own R.
        """
        self.sigma_a = sigma_a
        self.R = np.asarray(measurement_noise, dtype=float)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])   # we only measure position
        self.x = None
        self.P = None
        self.last_time = None
        self.n_updates = 0          # accepted updates since initialize (for the warmup)
        self.n_rejected = 0         # rejections in a row (for the lockout reset)

    def initialize(self, first_measurement, timestamp, R=None):
        """Start the filter from the first 3D measurement.

        R: optional 3×3 noise for this measurement. Stereo noise depends on
        distance, so each measurement can bring its own R; if None, the
        fixed self.R from __init__ is used.
        """
        R = self.R if R is None else R
        self.x = np.concatenate([first_measurement, np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = R                     # position: as good as one measurement
        self.P[3:, 3:] = 100.0 * np.eye(3)     # velocity: unknown (σ = 10 m/s)
        self.last_time = timestamp
        self.n_updates = 0
        self.n_rejected = 0

    def predict(self, dt):
        """Propagate the state and covariance forward by dt under gravity."""
        F = transition_matrix(dt)
        self.x = F @ self.x + gravity_input(dt)
        self.P = F @ self.P @ F.T + process_noise(dt, self.sigma_a)

    def update(self, measurement, R=None):
        """Correct the state with a new 3D position measurement.

        R: optional 3×3 noise for this measurement (see initialize).
        """
        R = self.R if R is None else R
        y = measurement - self.H @ self.x            # residual --> measured minus predicted
        S = self.H @ self.P @ self.H.T + R           # expected size of that residual

        # reject measurements implausibly far from the prediction,
        # but only once the filter has had enough updates to trust its prediction
        warmed_up = self.n_updates >= GATE_WARMUP_UPDATES
        if warmed_up and y @ np.linalg.solve(S, y) > GATE_THRESHOLD:
            self.n_rejected += 1
            if self.n_rejected >= MAX_CONSECUTIVE_REJECTIONS:
                # lockout: restart from this measurement (velocity unknown again)
                self.initialize(measurement, self.last_time, R)
            return False

        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y

        # Joseph form --> same as (I - KH) P but stays symmetric and positive
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.n_updates += 1
        self.n_rejected = 0
        return True

    def get_state(self):
        """Return the current state estimate and covariance."""
        return self.x.copy(), self.P.copy()

    def step(self, measurement, timestamp, R=None):
        """Predict to this timestamp, then update. Handles uneven frame gaps."""
        self.predict(timestamp - self.last_time)
        self.last_time = timestamp
        return self.update(measurement, R)


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
