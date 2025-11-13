"""
Regime-Switching Models for detecting structural breaks and transitions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


class RegimeSwitchingModel:
    """
    Markov Regime-Switching Model.

    Models systems that switch between different regimes (e.g., peace/war,
    stable/unstable) with different dynamics in each regime.
    """

    def __init__(self, n_regimes: int, n_features: int):
        """
        Initialize regime-switching model.

        Parameters
        ----------
        n_regimes : int
            Number of regimes
        n_features : int
            Number of features
        """
        self.n_regimes = n_regimes
        self.n_features = n_features

        # Regime-specific parameters
        self.means = np.random.randn(n_regimes, n_features)
        self.covariances = np.array([np.eye(n_features) for _ in range(n_regimes)])

        # Transition matrix
        self.transition_matrix = np.random.dirichlet(np.ones(n_regimes), size=n_regimes)

    def set_parameters(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        transition_matrix: np.ndarray
    ) -> None:
        """
        Set model parameters.

        Parameters
        ----------
        means : np.ndarray, shape (n_regimes, n_features)
            Mean for each regime
        covariances : np.ndarray, shape (n_regimes, n_features, n_features)
            Covariance for each regime
        transition_matrix : np.ndarray, shape (n_regimes, n_regimes)
            Regime transition probabilities
        """
        self.means = means
        self.covariances = covariances
        self.transition_matrix = transition_matrix

    def fit(self, data: np.ndarray, max_iter: int = 100) -> None:
        """
        Fit model using EM algorithm.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_features)
            Time series data
        max_iter : int
            Maximum EM iterations
        """
        n_samples = len(data)

        for iteration in range(max_iter):
            # E-step: compute regime probabilities
            regime_probs = self._compute_regime_probabilities(data)

            # M-step: update parameters
            for k in range(self.n_regimes):
                weights = regime_probs[:, k]
                total_weight = weights.sum()

                if total_weight > 0:
                    # Update mean
                    self.means[k] = np.sum(weights[:, np.newaxis] * data, axis=0) / total_weight

                    # Update covariance
                    diff = data - self.means[k]
                    self.covariances[k] = (weights[:, np.newaxis, np.newaxis] * \
                                          (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / total_weight

            # Update transition matrix
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    numerator = 0
                    denominator = 0
                    for t in range(n_samples - 1):
                        numerator += regime_probs[t, i] * regime_probs[t + 1, j]
                        denominator += regime_probs[t, i]

                    if denominator > 0:
                        self.transition_matrix[i, j] = numerator / denominator

            # Normalize transition matrix rows
            self.transition_matrix = self.transition_matrix / \
                                    self.transition_matrix.sum(axis=1, keepdims=True)

    def _compute_regime_probabilities(self, data: np.ndarray) -> np.ndarray:
        """
        Compute regime probabilities using filtering.

        Parameters
        ----------
        data : np.ndarray
            Data

        Returns
        -------
        np.ndarray
            Regime probabilities for each time step
        """
        n_samples = len(data)
        probs = np.zeros((n_samples, self.n_regimes))

        # Compute likelihoods
        likelihoods = np.zeros((n_samples, self.n_regimes))
        for k in range(self.n_regimes):
            likelihoods[:, k] = stats.multivariate_normal.pdf(
                data,
                mean=self.means[k],
                cov=self.covariances[k]
            )

        # Forward filtering
        probs[0] = likelihoods[0]
        probs[0] /= probs[0].sum()

        for t in range(1, n_samples):
            probs[t] = likelihoods[t] * (probs[t-1] @ self.transition_matrix)
            probs[t] /= probs[t].sum()

        return probs

    def predict_regime(self, data: np.ndarray) -> np.ndarray:
        """
        Predict most likely regime at each time step.

        Parameters
        ----------
        data : np.ndarray
            Time series data

        Returns
        -------
        np.ndarray
            Most likely regime at each time step
        """
        probs = self._compute_regime_probabilities(data)
        return np.argmax(probs, axis=1)

    def detect_regime_shifts(
        self,
        data: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> List[Dict[str, any]]:
        """
        Detect regime shifts in data.

        Parameters
        ----------
        data : np.ndarray
            Time series data
        confidence_threshold : float
            Minimum confidence for regime shift

        Returns
        -------
        list
            List of detected shifts
        """
        regimes = self.predict_regime(data)
        probs = self._compute_regime_probabilities(data)

        shifts = []
        for t in range(1, len(regimes)):
            if regimes[t] != regimes[t-1]:
                confidence = probs[t, regimes[t]]
                if confidence >= confidence_threshold:
                    shifts.append({
                        'time': t,
                        'from_regime': regimes[t-1],
                        'to_regime': regimes[t],
                        'confidence': confidence
                    })

        return shifts

    def forecast(
        self,
        current_regime: int,
        n_steps: int,
        n_simulations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future states using Monte Carlo.

        Parameters
        ----------
        current_regime : int
            Current regime
        n_steps : int
            Forecast horizon
        n_simulations : int
            Number of simulations

        Returns
        -------
        tuple
            (forecasts, regime_paths)
        """
        forecasts = np.zeros((n_simulations, n_steps, self.n_features))
        regime_paths = np.zeros((n_simulations, n_steps), dtype=int)

        for sim in range(n_simulations):
            regime = current_regime

            for t in range(n_steps):
                # Generate observation from current regime
                forecasts[sim, t] = np.random.multivariate_normal(
                    self.means[regime],
                    self.covariances[regime]
                )
                regime_paths[sim, t] = regime

                # Transition to next regime
                regime = np.random.choice(
                    self.n_regimes,
                    p=self.transition_matrix[regime]
                )

        return forecasts, regime_paths
