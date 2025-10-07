# ===================================================================
# Bio ANNa - Bayesian Fusion Engine
#
# Author: chandan sheikder
# Date: 07 oct 2025
#
# Description:
# This module contains the core state estimation logic for Project ANNa.
# It implements an Extended Kalman Filter (EKF) to fuse asynchronous
# measurements from the AntBot SNN (velocity) and the GridCore SNN
# (absolute pose corrections).
#
# The EKF maintains an estimate of the robot's state and its uncertainty,
# continuously refining it as new sensor data arrives.
#
# State Vector (x): [x, y, theta, v, w]
#   - x, y: Global position in the odom frame
#   - theta: Orientation (yaw) in the odom frame
#   - v: Forward linear velocity
#   - w: Angular velocity (yaw rate)
# ===================================================================

from typing import Optional

import numpy as np

class BayesianFusionEngine:
    """
    Implements an Extended Kalman Filter to fuse odometry and pose correction data.
    """
    def __init__(
        self,
        initial_state=None,
        initial_covariance=None,
        process_noise_variances=None,
        antbot_noise_variances=None,
        gridcore_noise_variances=None
    ):
        """
        Initializes the Extended Kalman Filter.
        Args:
            initial_state (np.array): The initial 5x1 state vector [x, y, theta, v, w].
            initial_covariance (np.array): The initial 5x5 covariance matrix (P).
            process_noise_variances (list): Variances for the process noise matrix (Q).
            antbot_noise_variances (list): Variances for the AntBot measurement noise (R_antbot).
            gridcore_noise_variances (list): Variances for the GridCore measurement noise (R_gridcore).
        """
        # Provide safe defaults to make the engine easy to instantiate.
        if initial_state is None:
            initial_state = np.zeros(5, dtype=float)
        if initial_covariance is None:
            initial_covariance = np.diag([0.5, 0.5, 0.1, 0.5, 0.5])
        if process_noise_variances is None:
            process_noise_variances = np.array([0.01, 0.01, 0.003, 0.05, 0.05], dtype=float)
        if antbot_noise_variances is None:
            antbot_noise_variances = np.array([0.1, 0.1], dtype=float)
        if gridcore_noise_variances is None:
            gridcore_noise_variances = np.array([0.2, 0.2, 0.05], dtype=float)

        # State vector [x, y, theta, v, w]
        self.x = np.asarray(initial_state, dtype=float).reshape(5, 1)
        # State covariance matrix
        initial_covariance = np.asarray(initial_covariance, dtype=float)
        if initial_covariance.shape == (5,):
            initial_covariance = np.diag(initial_covariance)
        self.P = initial_covariance.reshape(5, 5)
        if self.P.shape != (5, 5):
            raise ValueError("Initial covariance must be 5x5 or derived from 5 variances.")
        
        # Process noise covariance matrix (Q)
        # Represents uncertainty in our motion model (e.g., wheel slip, uneven ground).
        process_noise_variances = np.asarray(process_noise_variances, dtype=float)
        if process_noise_variances.ndim == 2:
            self.Q = process_noise_variances.reshape(5, 5)
        else:
            self.Q = np.diag(process_noise_variances)
        if self.Q.shape != (5, 5):
            raise ValueError("Process noise matrix must be 5x5 or derived from 5 variances.")
        
        # Measurement noise covariance matrices (R)
        # Represents how much we trust our sensors.
        antbot_noise_variances = np.asarray(antbot_noise_variances, dtype=float)
        if antbot_noise_variances.ndim == 2:
            self.R_antbot = antbot_noise_variances.reshape(2, 2)
        else:
            self.R_antbot = np.diag(antbot_noise_variances)
        if self.R_antbot.shape != (2, 2):
            raise ValueError("AntBot measurement noise must be 2x2 or derived from 2 variances.")

        gridcore_noise_variances = np.asarray(gridcore_noise_variances, dtype=float)
        if gridcore_noise_variances.ndim == 2:
            self.R_gridcore = gridcore_noise_variances.reshape(3, 3)
        else:
            self.R_gridcore = np.diag(gridcore_noise_variances)
        if self.R_gridcore.shape != (3, 3):
            raise ValueError("GridCore measurement noise must be 3x3 or derived from 3 variances.")

    def _normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def predict(self, dt: float):
        """
        EKF Predict Step. Projects the state and covariance forward in time.
        This is based on the robot's motion model.
        """
        # --- 1. Project the state forward ---
        x, y, theta, v, w = self.x.flatten()
        
        # Non-linear motion model:
        # If angular velocity is very small, motion is linear
        if abs(w) < 1e-6:
            dx = v * np.cos(theta)
            dy = v * np.sin(theta)
        else:
            # Otherwise, motion is an arc
            radius = v / w
            dx = radius * (-np.sin(theta) + np.sin(theta + w * dt))
            dy = radius * ( np.cos(theta) - np.cos(theta + w * dt))
            
        self.x[0] += dx
        self.x[1] += dy
        self.x[2] += w * dt
        # Velocities are assumed to persist unless updated by a measurement
        
        self.x[2] = self._normalize_angle(self.x[2])

        # --- 2. Calculate the Jacobian of the motion model (F) ---
        # F is the partial derivative of the motion model w.r.t the state.
        F = np.eye(5)
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        F[2, 4] = dt
        
        # --- 3. Project the covariance forward ---
        # P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_with_antbot_odometry(self, measurement: np.ndarray, measurement_covariance: Optional[np.ndarray] = None):
        """
        EKF Update Step for a velocity measurement from the AntBot SNN.
        Args:
            measurement (np.array): A 2x1 vector [v, w].
            measurement_covariance (np.array, optional): Optional 2x2 covariance matrix.
        """
        measurement = measurement.reshape(2, 1)
        
        # --- 1. Define the measurement model Jacobian (H) ---
        # H maps the state space to the measurement space.
        # We are measuring v and w, which are the 4th and 5th elements of the state.
        H = np.array([
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        
        # --- 2. Perform the Kalman Update ---
        R = self.R_antbot if measurement_covariance is None else np.asarray(measurement_covariance, dtype=float).reshape(2, 2)
        self._perform_update(measurement, H, R)

    def update_with_gridcore_correction(self, measurement: np.ndarray, measurement_covariance: Optional[np.ndarray] = None):
        """
        EKF Update Step for an absolute pose measurement from the GridCore SNN.
        Args:
            measurement (np.array): A 3x1 vector [x, y, theta].
            measurement_covariance (np.array, optional): Optional 3x3 covariance matrix.
        """
        measurement = measurement.reshape(3, 1)
        
        # --- 1. Define the measurement model Jacobian (H) ---
        # We are measuring x, y, and theta, the first 3 elements of the state.
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])
        
        # --- 2. Perform the Kalman Update ---
        # Note: A more advanced version could use the covariance from the
        #       GridCore message to dynamically set R_gridcore here.
        R = self.R_gridcore if measurement_covariance is None else np.asarray(measurement_covariance, dtype=float).reshape(3, 3)
        self._perform_update(measurement, H, R)

    def _perform_update(self, z, H, R):
        """
        The generic Kalman update logic.
        Args:
            z (np.array): The measurement vector.
            H (np.array): The measurement Jacobian.
            R (np.array): The measurement noise covariance.
        """
        # --- 1. Calculate the measurement residual (y) ---
        # y = z - H * x
        # The residual for the angle needs special handling (normalization)
        z_predicted = H @ self.x
        y = z - z_predicted
        if H.shape[0] == 3: # If this is a pose update
            y[2] = self._normalize_angle(y[2])
        
        # --- 2. Calculate the residual covariance (S) ---
        # S = H * P * H^T + R
        S = H @ self.P @ H.T + R
        
        # --- 3. Calculate the Kalman Gain (K) ---
        # K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # --- 4. Update the state estimate ---
        # x = x + K * y
        self.x = self.x + K @ y
        self.x[2] = self._normalize_angle(self.x[2])
        
        # --- 5. Update the state covariance ---
        # P = (I - K * H) * P
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P

    def get_current_pose(self) -> np.ndarray:
        """Returns the fused [x, y, theta] from the state vector."""
        return self.x[0:3].flatten()

    def get_current_velocity(self) -> np.ndarray:
        """Returns the fused [v, w] from the state vector."""
        return self.x[3:5].flatten()

    def get_current_pose_covariance(self) -> np.ndarray:
        """Returns the 3x3 covariance matrix for the pose [x, y, theta]."""
        return self.P[0:3, 0:3]
