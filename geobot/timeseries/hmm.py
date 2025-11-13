"""
Hidden Markov Model for regime detection and state estimation.
"""

import numpy as np
from typing import Optional, Tuple, List


class HiddenMarkovModel:
    """
    Hidden Markov Model for detecting regime changes.

    Useful for identifying transitions between:
    - Peace <-> Conflict
    - Stable <-> Unstable
    - Cooperative <-> Hostile
    """

    def __init__(
        self,
        n_states: int,
        n_observations: int
    ):
        """
        Initialize HMM.

        Parameters
        ----------
        n_states : int
            Number of hidden states
        n_observations : int
            Number of possible observations
        """
        self.n_states = n_states
        self.n_observations = n_observations

        # Initialize parameters randomly
        self.transition_matrix = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.emission_matrix = np.random.dirichlet(np.ones(n_observations), size=n_states)
        self.initial_probs = np.ones(n_states) / n_states

    def set_parameters(
        self,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_probs: np.ndarray
    ) -> None:
        """
        Set HMM parameters.

        Parameters
        ----------
        transition_matrix : np.ndarray, shape (n_states, n_states)
            State transition probabilities
        emission_matrix : np.ndarray, shape (n_states, n_observations)
            Observation emission probabilities
        initial_probs : np.ndarray, shape (n_states,)
            Initial state probabilities
        """
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_probs = initial_probs

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm for computing state probabilities.

        Parameters
        ----------
        observations : np.ndarray
            Sequence of observations

        Returns
        -------
        tuple
            (forward_probabilities, log_likelihood)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialize
        alpha[0] = self.initial_probs * self.emission_matrix[:, observations[0]]
        alpha[0] /= alpha[0].sum()

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j]) * \
                             self.emission_matrix[j, observations[t]]
            alpha[t] /= alpha[t].sum()  # Normalize to prevent underflow

        log_likelihood = np.sum(np.log(alpha.sum(axis=1)))

        return alpha, log_likelihood

    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm for most likely state sequence.

        Parameters
        ----------
        observations : np.ndarray
            Sequence of observations

        Returns
        -------
        tuple
            (most_likely_states, log_probability)
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        delta[0] = np.log(self.initial_probs) + \
                   np.log(self.emission_matrix[:, observations[0]] + 1e-10)

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + \
                             np.log(self.emission_matrix[j, observations[t]] + 1e-10)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        log_prob = np.max(delta[-1])

        return states, log_prob

    def detect_regime_change(
        self,
        observations: np.ndarray,
        threshold: float = 0.7
    ) -> List[int]:
        """
        Detect regime changes in observation sequence.

        Parameters
        ----------
        observations : np.ndarray
            Observations
        threshold : float
            Confidence threshold for regime change

        Returns
        -------
        list
            Indices where regime changes occurred
        """
        states, _ = self.viterbi(observations)

        changes = []
        for t in range(1, len(states)):
            if states[t] != states[t-1]:
                changes.append(t)

        return changes
