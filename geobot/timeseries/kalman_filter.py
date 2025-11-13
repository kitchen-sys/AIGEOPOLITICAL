"""
Kalman Filter and Extended Kalman Filter for State-Space Models

Detect:
- When a regime is becoming unstable
- When a country is shifting to wartime posture
- When propaganda spikes precede unrest
- When mobilization signals are present
"""

import numpy as np
from typing import Optional, Tuple, Callable


class KalmanFilter:
    """
    Linear Kalman Filter for state-space estimation.

    State-space model:
    x_{t+1} = F @ x_t + B @ u_t + w_t  (state transition)
    y_t = H @ x_t + v_t                (observation)

    where:
    - x_t is the hidden state
    - y_t is the observation
    - u_t is control input
    - w_t ~ N(0, Q) is process noise
    - v_t ~ N(0, R) is observation noise
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        B: Optional[np.ndarray] = None
    ):
        """
        Initialize Kalman Filter.

        Parameters
        ----------
        F : np.ndarray
            State transition matrix
        H : np.ndarray
            Observation matrix
        Q : np.ndarray
            Process noise covariance
        R : np.ndarray
            Observation noise covariance
        B : np.ndarray, optional
            Control input matrix
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B

        self.n_states = F.shape[0]
        self.n_obs = H.shape[0]

        # Initialize state and covariance
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states)

        # History
        self.history = {
            'x': [],
            'P': [],
            'K': [],  # Kalman gain
            'innovation': []
        }

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step.

        Parameters
        ----------
        u : np.ndarray, optional
            Control input

        Returns
        -------
        tuple
            (predicted_state, predicted_covariance)
        """
        # Predict state
        if u is not None and self.B is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x, self.P

    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step given observation.

        Parameters
        ----------
        y : np.ndarray
            Observation

        Returns
        -------
        tuple
            (updated_state, updated_covariance)
        """
        # Innovation
        y_pred = self.H @ self.x
        innovation = y - y_pred

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ innovation

        # Update covariance
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P

        # Save to history
        self.history['x'].append(self.x.copy())
        self.history['P'].append(self.P.copy())
        self.history['K'].append(K.copy())
        self.history['innovation'].append(innovation.copy())

        return self.x, self.P

    def filter(
        self,
        observations: np.ndarray,
        controls: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run filter on sequence of observations.

        Parameters
        ----------
        observations : np.ndarray, shape (n_timesteps, n_obs)
            Sequence of observations
        controls : np.ndarray, optional, shape (n_timesteps, n_controls)
            Sequence of control inputs

        Returns
        -------
        tuple
            (states, covariances)
        """
        n_timesteps = observations.shape[0]
        states = np.zeros((n_timesteps, self.n_states))
        covariances = np.zeros((n_timesteps, self.n_states, self.n_states))

        for t in range(n_timesteps):
            # Predict
            u = controls[t] if controls is not None else None
            self.predict(u)

            # Update
            self.update(observations[t])

            states[t] = self.x
            covariances[t] = self.P

        return states, covariances

    def smooth(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rauch-Tung-Striebel smoother (backward pass).

        Parameters
        ----------
        observations : np.ndarray
            Sequence of observations

        Returns
        -------
        tuple
            (smoothed_states, smoothed_covariances)
        """
        # Forward pass
        states_fwd, covs_fwd = self.filter(observations)

        n_timesteps = len(observations)
        states_smooth = np.zeros_like(states_fwd)
        covs_smooth = np.zeros_like(covs_fwd)

        # Initialize with last filtered estimate
        states_smooth[-1] = states_fwd[-1]
        covs_smooth[-1] = covs_fwd[-1]

        # Backward pass
        for t in range(n_timesteps - 2, -1, -1):
            # Predict forward
            x_pred = self.F @ states_fwd[t]
            P_pred = self.F @ covs_fwd[t] @ self.F.T + self.Q

            # Smoother gain
            J = covs_fwd[t] @ self.F.T @ np.linalg.inv(P_pred)

            # Smooth
            states_smooth[t] = states_fwd[t] + J @ (states_smooth[t + 1] - x_pred)
            covs_smooth[t] = covs_fwd[t] + J @ (covs_smooth[t + 1] - P_pred) @ J.T

        return states_smooth, covs_smooth

    def detect_anomaly(self, innovation_threshold: float = 3.0) -> bool:
        """
        Detect anomaly based on innovation magnitude.

        Parameters
        ----------
        innovation_threshold : float
            Threshold in standard deviations

        Returns
        -------
        bool
            True if anomaly detected
        """
        if len(self.history['innovation']) == 0:
            return False

        innovation = self.history['innovation'][-1]
        S = self.H @ self.P @ self.H.T + self.R

        # Normalized innovation
        normalized = innovation.T @ np.linalg.inv(S) @ innovation

        return normalized > innovation_threshold ** 2


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state-space models.

    Nonlinear model:
    x_{t+1} = f(x_t, u_t) + w_t
    y_t = h(x_t) + v_t
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        F_jacobian: Callable,
        H_jacobian: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        n_states: int,
        n_obs: int
    ):
        """
        Initialize Extended Kalman Filter.

        Parameters
        ----------
        f : callable
            State transition function
        h : callable
            Observation function
        F_jacobian : callable
            Jacobian of f
        H_jacobian : callable
            Jacobian of h
        Q : np.ndarray
            Process noise covariance
        R : np.ndarray
            Observation noise covariance
        n_states : int
            Number of state variables
        n_obs : int
            Number of observations
        """
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.n_states = n_states
        self.n_obs = n_obs

        # Initialize
        self.x = np.zeros(n_states)
        self.P = np.eye(n_states)

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step.

        Parameters
        ----------
        u : np.ndarray, optional
            Control input

        Returns
        -------
        tuple
            (predicted_state, predicted_covariance)
        """
        # Predict state (nonlinear)
        self.x = self.f(self.x, u)

        # Linearize at current state
        F = self.F_jacobian(self.x, u)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

        return self.x, self.P

    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step.

        Parameters
        ----------
        y : np.ndarray
            Observation

        Returns
        -------
        tuple
            (updated_state, updated_covariance)
        """
        # Predict observation
        y_pred = self.h(self.x)
        innovation = y - y_pred

        # Linearize observation model
        H = self.H_jacobian(self.x)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ innovation

        # Update covariance
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P

        return self.x, self.P

    def filter(
        self,
        observations: np.ndarray,
        controls: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run filter on observation sequence.

        Parameters
        ----------
        observations : np.ndarray
            Observations
        controls : np.ndarray, optional
            Control inputs

        Returns
        -------
        tuple
            (states, covariances)
        """
        n_timesteps = observations.shape[0]
        states = np.zeros((n_timesteps, self.n_states))
        covariances = np.zeros((n_timesteps, self.n_states, self.n_states))

        for t in range(n_timesteps):
            u = controls[t] if controls is not None else None
            self.predict(u)
            self.update(observations[t])

            states[t] = self.x
            covariances[t] = self.P

        return states, covariances
