"""
Sequential Monte Carlo (SMC) and Particle Filtering

Implements advanced particle filtering algorithms for:
- Recursive Bayesian inference on latent states
- High-dimensional posterior computation
- Nonlinear/non-Gaussian state estimation
- Degeneracy handling through resampling

Methods:
- Bootstrap particle filter
- Auxiliary particle filter
- Rao-Blackwellized particle filter
- Systematic resampling, stratified resampling
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.stats import multivariate_normal


@dataclass
class ParticleState:
    """
    Represents a particle filter state.

    Attributes
    ----------
    particles : np.ndarray, shape (n_particles, state_dim)
        Particle positions
    weights : np.ndarray, shape (n_particles,)
        Normalized particle weights
    log_weights : np.ndarray, shape (n_particles,)
        Log weights (for numerical stability)
    ess : float
        Effective sample size
    """
    particles: np.ndarray
    weights: np.ndarray
    log_weights: np.ndarray
    ess: float


class SequentialMonteCarlo:
    """
    Sequential Monte Carlo (particle filter) for recursive Bayesian inference.

    Performs filtering on nonlinear/non-Gaussian state-space models:
    x_t ~ p(x_t | x_{t-1})  (dynamics)
    y_t ~ p(y_t | x_t)      (observation)

    Maintains posterior approximation p(x_t | y_{1:t}) via weighted particles.
    """

    def __init__(
        self,
        n_particles: int,
        state_dim: int,
        dynamics_fn: Callable,
        observation_fn: Callable,
        dynamics_noise_fn: Optional[Callable] = None,
        observation_noise_fn: Optional[Callable] = None,
        resample_threshold: float = 0.5
    ):
        """
        Initialize Sequential Monte Carlo filter.

        Parameters
        ----------
        n_particles : int
            Number of particles
        state_dim : int
            Dimension of state space
        dynamics_fn : callable
            State transition function: x_t = f(x_{t-1}, noise)
        observation_fn : callable
            Observation likelihood: p(y_t | x_t)
        dynamics_noise_fn : callable, optional
            Dynamics noise sampler
        observation_noise_fn : callable, optional
            Observation noise sampler
        resample_threshold : float
            ESS threshold for resampling (as fraction of n_particles)
        """
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.dynamics_fn = dynamics_fn
        self.observation_fn = observation_fn
        self.dynamics_noise_fn = dynamics_noise_fn
        self.observation_noise_fn = observation_noise_fn
        self.resample_threshold = resample_threshold

        # Initialize particles uniformly (or from prior)
        self.particles = np.random.randn(n_particles, state_dim)
        self.weights = np.ones(n_particles) / n_particles
        self.log_weights = np.log(self.weights)

        # History
        self.history = []

    def initialize_from_prior(self, prior_sampler: Callable) -> None:
        """
        Initialize particles from prior distribution.

        Parameters
        ----------
        prior_sampler : callable
            Function that samples from prior: x ~ p(x_0)
        """
        self.particles = np.array([prior_sampler() for _ in range(self.n_particles)])
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.log_weights = np.log(self.weights)

    def predict(self) -> None:
        """
        Prediction step: propagate particles through dynamics.

        x_t^i ~ p(x_t | x_{t-1}^i)
        """
        new_particles = np.zeros_like(self.particles)

        for i in range(self.n_particles):
            # Sample noise
            if self.dynamics_noise_fn:
                noise = self.dynamics_noise_fn()
            else:
                noise = np.random.randn(self.state_dim) * 0.1

            # Propagate particle
            new_particles[i] = self.dynamics_fn(self.particles[i], noise)

        self.particles = new_particles

    def update(self, observation: np.ndarray) -> None:
        """
        Update step: reweight particles based on observation likelihood.

        w_t^i ∝ p(y_t | x_t^i) w_{t-1}^i

        Parameters
        ----------
        observation : np.ndarray
            Observation y_t
        """
        # Compute log-likelihoods
        log_likelihoods = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            log_likelihoods[i] = self.observation_fn(observation, self.particles[i])

        # Update log-weights
        self.log_weights = self.log_weights + log_likelihoods

        # Normalize weights (in log space for stability)
        max_log_weight = np.max(self.log_weights)
        self.log_weights = self.log_weights - max_log_weight
        self.weights = np.exp(self.log_weights)
        self.weights = self.weights / np.sum(self.weights)
        self.log_weights = np.log(self.weights)

    def compute_ess(self) -> float:
        """
        Compute effective sample size (ESS).

        ESS = 1 / sum(w_i^2)

        Returns
        -------
        float
            Effective sample size
        """
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self, method: str = 'systematic') -> None:
        """
        Resample particles to combat degeneracy.

        Parameters
        ----------
        method : str
            Resampling method ('systematic', 'stratified', 'multinomial')
        """
        if method == 'systematic':
            indices = self._systematic_resample()
        elif method == 'stratified':
            indices = self._stratified_resample()
        elif method == 'multinomial':
            indices = np.random.choice(
                self.n_particles,
                size=self.n_particles,
                p=self.weights
            )
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        # Resample particles
        self.particles = self.particles[indices]

        # Reset weights to uniform
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.log_weights = np.log(self.weights)

    def _systematic_resample(self) -> np.ndarray:
        """
        Systematic resampling (low variance).

        Returns
        -------
        np.ndarray
            Resampled indices
        """
        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indices = np.zeros(self.n_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)

        i, j = 0, 0
        while i < self.n_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def _stratified_resample(self) -> np.ndarray:
        """
        Stratified resampling.

        Returns
        -------
        np.ndarray
            Resampled indices
        """
        positions = (np.arange(self.n_particles) + np.random.uniform(size=self.n_particles)) / self.n_particles
        indices = np.zeros(self.n_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)

        i, j = 0, 0
        while i < self.n_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def filter_step(self, observation: np.ndarray, resample: bool = True) -> ParticleState:
        """
        Single filtering step: predict + update + (optional) resample.

        Parameters
        ----------
        observation : np.ndarray
            Observation at current time
        resample : bool
            Whether to check ESS and resample if needed

        Returns
        -------
        ParticleState
            Current particle filter state
        """
        # Predict
        self.predict()

        # Update
        self.update(observation)

        # Compute ESS
        ess = self.compute_ess()

        # Resample if ESS too low
        if resample and ess < self.resample_threshold * self.n_particles:
            self.resample(method='systematic')
            ess = self.n_particles  # After resampling, ESS = N

        # Save state
        state = ParticleState(
            particles=self.particles.copy(),
            weights=self.weights.copy(),
            log_weights=self.log_weights.copy(),
            ess=ess
        )
        self.history.append(state)

        return state

    def filter(self, observations: np.ndarray) -> list:
        """
        Run particle filter on sequence of observations.

        Parameters
        ----------
        observations : np.ndarray, shape (n_timesteps, obs_dim)
            Sequence of observations

        Returns
        -------
        list
            List of ParticleState objects
        """
        states = []
        for obs in observations:
            state = self.filter_step(obs)
            states.append(state)
        return states

    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get posterior mean and covariance estimate.

        Returns
        -------
        tuple
            (mean, covariance)
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)

        # Weighted covariance
        diff = self.particles - mean
        cov = np.dot(self.weights * diff.T, diff)

        return mean, cov


class AuxiliaryParticleFilter(SequentialMonteCarlo):
    """
    Auxiliary Particle Filter.

    Improves importance distribution by looking ahead at next observation.
    Uses auxiliary variables to guide particle propagation.
    """

    def __init__(self, *args, look_ahead_fn: Optional[Callable] = None, **kwargs):
        """
        Initialize auxiliary particle filter.

        Parameters
        ----------
        look_ahead_fn : callable, optional
            Function to compute look-ahead weights: μ_t^i = p(y_t | m_t^i)
            where m_t^i is a prediction of x_t from x_{t-1}^i
        """
        super().__init__(*args, **kwargs)
        self.look_ahead_fn = look_ahead_fn

    def filter_step(self, observation: np.ndarray, resample: bool = True) -> ParticleState:
        """
        Auxiliary particle filter step.

        Parameters
        ----------
        observation : np.ndarray
            Current observation
        resample : bool
            Whether to resample

        Returns
        -------
        ParticleState
            Filter state
        """
        # Step 1: Compute auxiliary weights (look-ahead)
        if self.look_ahead_fn:
            aux_weights = np.zeros(self.n_particles)
            for i in range(self.n_particles):
                # Predict particle position
                predicted = self.dynamics_fn(self.particles[i], np.zeros(self.state_dim))
                # Compute look-ahead likelihood
                aux_weights[i] = np.exp(self.observation_fn(observation, predicted))

            # Combine with current weights
            aux_weights = self.weights * aux_weights
            aux_weights = aux_weights / np.sum(aux_weights)
        else:
            aux_weights = self.weights

        # Step 2: Resample using auxiliary weights
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=aux_weights)
        self.particles = self.particles[indices]
        selected_weights = self.weights[indices]
        selected_aux_weights = aux_weights[indices]

        # Step 3: Propagate
        self.predict()

        # Step 4: Update with importance weights
        self.update(observation)

        # Adjust weights for auxiliary sampling
        self.weights = self.weights * selected_weights / (selected_aux_weights + 1e-10)
        self.weights = self.weights / np.sum(self.weights)
        self.log_weights = np.log(self.weights)

        # ESS and resampling
        ess = self.compute_ess()
        if resample and ess < self.resample_threshold * self.n_particles:
            self.resample()
            ess = self.n_particles

        state = ParticleState(
            particles=self.particles.copy(),
            weights=self.weights.copy(),
            log_weights=self.log_weights.copy(),
            ess=ess
        )
        self.history.append(state)

        return state


class RaoBlackwellizedParticleFilter:
    """
    Rao-Blackwellized Particle Filter (RBPF).

    For models with linear-Gaussian substructure:
    - Part of state updated with Kalman filter (exact)
    - Remaining part updated with particle filter

    Reduces variance by marginalizing out linear components.
    """

    def __init__(
        self,
        n_particles: int,
        nonlinear_dim: int,
        linear_dim: int,
        nonlinear_dynamics_fn: Callable,
        linear_dynamics_fn: Callable,
        observation_fn: Callable,
        F_linear: np.ndarray,
        H_linear: np.ndarray,
        Q_linear: np.ndarray,
        R: np.ndarray
    ):
        """
        Initialize Rao-Blackwellized particle filter.

        Parameters
        ----------
        n_particles : int
            Number of particles for nonlinear part
        nonlinear_dim : int
            Dimension of nonlinear state
        linear_dim : int
            Dimension of linear state
        nonlinear_dynamics_fn : callable
            Nonlinear state dynamics
        linear_dynamics_fn : callable
            Linear state dynamics (conditioned on nonlinear state)
        observation_fn : callable
            Observation likelihood
        F_linear : np.ndarray
            Linear dynamics matrix
        H_linear : np.ndarray
            Linear observation matrix
        Q_linear : np.ndarray
            Linear process noise covariance
        R : np.ndarray
            Observation noise covariance
        """
        self.n_particles = n_particles
        self.nonlinear_dim = nonlinear_dim
        self.linear_dim = linear_dim

        self.nonlinear_dynamics_fn = nonlinear_dynamics_fn
        self.linear_dynamics_fn = linear_dynamics_fn
        self.observation_fn = observation_fn

        # Linear substructure parameters (for Kalman filter)
        self.F_linear = F_linear
        self.H_linear = H_linear
        self.Q_linear = Q_linear
        self.R = R

        # Initialize particles (nonlinear part)
        self.nonlinear_particles = np.random.randn(n_particles, nonlinear_dim)
        self.weights = np.ones(n_particles) / n_particles

        # Initialize Kalman filters (one per particle)
        self.linear_means = [np.zeros(linear_dim) for _ in range(n_particles)]
        self.linear_covs = [np.eye(linear_dim) for _ in range(n_particles)]

    def filter_step(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        RBPF filtering step.

        Parameters
        ----------
        observation : np.ndarray
            Current observation

        Returns
        -------
        tuple
            (nonlinear_estimate, linear_estimate)
        """
        # Step 1: Propagate nonlinear particles
        new_nonlinear_particles = np.zeros_like(self.nonlinear_particles)
        for i in range(self.n_particles):
            noise = np.random.randn(self.nonlinear_dim) * 0.1
            new_nonlinear_particles[i] = self.nonlinear_dynamics_fn(
                self.nonlinear_particles[i], noise
            )

        # Step 2: Update linear state with Kalman filter (per particle)
        new_linear_means = []
        new_linear_covs = []
        log_likelihoods = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            # Kalman prediction
            m_pred = self.F_linear @ self.linear_means[i]
            P_pred = self.F_linear @ self.linear_covs[i] @ self.F_linear.T + self.Q_linear

            # Kalman update
            innovation = observation - self.H_linear @ m_pred
            S = self.H_linear @ P_pred @ self.H_linear.T + self.R
            K = P_pred @ self.H_linear.T @ np.linalg.inv(S)

            m_new = m_pred + K @ innovation
            P_new = (np.eye(self.linear_dim) - K @ self.H_linear) @ P_pred

            new_linear_means.append(m_new)
            new_linear_covs.append(P_new)

            # Log-likelihood
            log_likelihoods[i] = multivariate_normal.logpdf(innovation, mean=np.zeros_like(innovation), cov=S)

        # Step 3: Update weights
        max_ll = np.max(log_likelihoods)
        weights = np.exp(log_likelihoods - max_ll)
        self.weights = self.weights * weights
        self.weights = self.weights / np.sum(self.weights)

        # Step 4: Resample if needed
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < 0.5 * self.n_particles:
            indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
            new_nonlinear_particles = new_nonlinear_particles[indices]
            new_linear_means = [new_linear_means[i] for i in indices]
            new_linear_covs = [new_linear_covs[i] for i in indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Update state
        self.nonlinear_particles = new_nonlinear_particles
        self.linear_means = new_linear_means
        self.linear_covs = new_linear_covs

        # Estimates
        nonlinear_estimate = np.average(self.nonlinear_particles, weights=self.weights, axis=0)
        linear_estimate = np.average(new_linear_means, weights=self.weights, axis=0)

        return nonlinear_estimate, linear_estimate
