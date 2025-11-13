"""
Hawkes Processes and Point Process Models for Conflict Modeling

Hawkes processes are self-exciting point processes where past events increase
the probability of future events. Critical for modeling:
- Conflict contagion and escalation dynamics
- Terrorist attack clustering
- Diplomatic incident cascades
- Arms race dynamics
- Protest contagion

Mathematical foundation:
λ(t) = μ + ∫_{-∞}^t φ(t - s) dN(s)

where:
- λ(t): instantaneous event rate (intensity)
- μ: baseline rate
- φ(t): excitation kernel (how past events affect current rate)
- N(s): counting process of past events

Key concepts:
- Branching ratio: Expected number of offspring events per parent
- If branching ratio < 1: process is stable (subcritical)
- If branching ratio ≥ 1: process is explosive (supercritical)
"""

import numpy as np
from scipy import optimize, stats, integrate
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
import warnings


@dataclass
class HawkesParameters:
    """Parameters for a Hawkes process."""
    mu: float  # Baseline intensity
    alpha: float  # Excitation amplitude
    beta: float  # Decay rate

    @property
    def branching_ratio(self) -> float:
        """Expected number of offspring per event."""
        return self.alpha / self.beta

    @property
    def is_stable(self) -> bool:
        """Check if process is subcritical (stable)."""
        return self.branching_ratio < 1.0


@dataclass
class HawkesFitResult:
    """Results from fitting a Hawkes process."""
    params: HawkesParameters
    log_likelihood: float
    aic: float
    bic: float
    n_events: int
    time_span: float
    intensity_trace: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None


class UnivariateHawkesProcess:
    """
    Univariate (1-dimensional) Hawkes Process.

    Intensity function:
    λ(t) = μ + α ∑_{t_i < t} exp(-β(t - t_i))

    This is a self-exciting process where each event increases future intensity.

    Example:
        >>> hawkes = UnivariateHawkesProcess()
        >>> events = hawkes.simulate(mu=0.5, alpha=0.8, beta=1.5, T=100.0)
        >>> result = hawkes.fit(events, T=100.0)
        >>> print(f"Branching ratio: {result.params.branching_ratio:.3f}")
        >>> prediction = hawkes.predict_intensity(events, result.params, t=105.0)
    """

    def __init__(self, kernel: str = 'exponential'):
        """
        Initialize Hawkes process.

        Args:
            kernel: Excitation kernel type ('exponential', 'power_law')
        """
        self.kernel = kernel

    def simulate(self, mu: float, alpha: float, beta: float, T: float,
                 max_events: int = 10000) -> np.ndarray:
        """
        Simulate Hawkes process using Ogata's thinning algorithm.

        Args:
            mu: Baseline intensity
            alpha: Excitation amplitude
            beta: Decay rate
            T: Time horizon
            max_events: Maximum number of events to generate

        Returns:
            Array of event times
        """
        events = []
        t = 0.0
        lambda_star = mu  # Upper bound on intensity

        while t < T and len(events) < max_events:
            # Generate candidate event
            lambda_star = self._compute_intensity(t, events, mu, alpha, beta)

            # Add safety margin
            lambda_star = lambda_star * 1.1 + 0.01

            # Draw inter-event time from exponential
            u = np.random.uniform()
            if lambda_star <= 0:
                break
            t = t - np.log(u) / lambda_star

            if t > T:
                break

            # Acceptance-rejection
            lambda_t = self._compute_intensity(t, events, mu, alpha, beta)
            D = np.random.uniform()

            if D * lambda_star <= lambda_t:
                events.append(t)

        return np.array(events)

    def fit(self, events: np.ndarray, T: float,
            initial_guess: Optional[Tuple[float, float, float]] = None) -> HawkesFitResult:
        """
        Fit Hawkes process parameters using maximum likelihood.

        Args:
            events: Array of event times
            T: Time horizon (observation period end)
            initial_guess: Initial parameter guess (mu, alpha, beta)

        Returns:
            HawkesFitResult with estimated parameters
        """
        events = np.asarray(events)
        events = np.sort(events)
        n_events = len(events)

        if initial_guess is None:
            # Initialize with reasonable defaults
            mu_init = n_events / T  # Average rate
            alpha_init = mu_init * 0.5  # Conservative excitation
            beta_init = 1.0
            initial_guess = (mu_init, alpha_init, beta_init)

        # Define negative log-likelihood
        def neg_log_likelihood(params):
            mu, alpha, beta = params

            # Constrain to positive values
            if mu <= 0 or alpha <= 0 or beta <= 0:
                return 1e10

            # Check stability
            if alpha / beta >= 1.0:
                return 1e10  # Penalize explosive processes

            return -self._log_likelihood(events, T, mu, alpha, beta)

        # Optimize
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]
        result = optimize.minimize(
            neg_log_likelihood,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        mu_opt, alpha_opt, beta_opt = result.x
        log_likelihood = -result.fun

        # Compute information criteria
        n_params = 3
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_events) * n_params

        params = HawkesParameters(mu=mu_opt, alpha=alpha_opt, beta=beta_opt)

        return HawkesFitResult(
            params=params,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_events=n_events,
            time_span=T
        )

    def predict_intensity(self, events: np.ndarray, params: HawkesParameters,
                         t: float) -> float:
        """
        Predict intensity at time t given past events.

        Args:
            events: Past event times (must be < t)
            params: Hawkes parameters
            t: Time to predict intensity

        Returns:
            Intensity λ(t)
        """
        return self._compute_intensity(t, events, params.mu, params.alpha, params.beta)

    def _compute_intensity(self, t: float, events: List[float],
                          mu: float, alpha: float, beta: float) -> float:
        """Compute intensity at time t."""
        if len(events) == 0:
            return mu

        events_array = np.asarray(events)
        past_events = events_array[events_array < t]

        if len(past_events) == 0:
            return mu

        # Exponential kernel
        excitation = alpha * np.sum(np.exp(-beta * (t - past_events)))

        return mu + excitation

    def _log_likelihood(self, events: np.ndarray, T: float,
                       mu: float, alpha: float, beta: float) -> float:
        """
        Compute log-likelihood for Hawkes process.

        LL = ∑_i log(λ(t_i)) - ∫_0^T λ(s) ds
        """
        n_events = len(events)

        if n_events == 0:
            return -mu * T

        # First term: ∑ log(λ(t_i))
        log_sum = 0.0
        for i, t_i in enumerate(events):
            lambda_i = self._compute_intensity(t_i, events[:i], mu, alpha, beta)
            if lambda_i <= 0:
                return -np.inf
            log_sum += np.log(lambda_i)

        # Second term: ∫_0^T λ(s) ds
        # For exponential kernel, this has closed form:
        # ∫_0^T λ(s) ds = μT + α ∑_i (1 - exp(-β(T - t_i))) / β

        integral = mu * T
        integral += alpha * np.sum(1 - np.exp(-beta * (T - events))) / beta

        return log_sum - integral


class MultivariateHawkesProcess:
    """
    Multivariate Hawkes Process for multiple interacting event streams.

    For K event types, the intensity of type k is:
    λ_k(t) = μ_k + ∑_{j=1}^K α_{kj} ∑_{t_i^j < t} φ_{kj}(t - t_i^j)

    This captures cross-excitation between different event types.
    Example: Conflict in country A affects conflict probability in country B.

    Example:
        >>> # Model 3 countries with mutual excitation
        >>> hawkes = MultivariateHawkesProcess(n_dimensions=3)
        >>> events = hawkes.simulate(
        ...     mu=np.array([0.5, 0.3, 0.4]),
        ...     alpha=np.array([[0.2, 0.1, 0.05],
        ...                     [0.15, 0.3, 0.1],
        ...                     [0.1, 0.1, 0.25]]),
        ...     beta=np.ones((3, 3)),
        ...     T=100.0
        ... )
        >>> result = hawkes.fit(events, T=100.0)
    """

    def __init__(self, n_dimensions: int, kernel: str = 'exponential'):
        """
        Initialize multivariate Hawkes process.

        Args:
            n_dimensions: Number of event types (dimensions)
            kernel: Excitation kernel type
        """
        self.n_dimensions = n_dimensions
        self.kernel = kernel

    def simulate(self, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                 T: float, max_events: int = 10000) -> List[List[float]]:
        """
        Simulate multivariate Hawkes process.

        Args:
            mu: Baseline intensities, shape (K,)
            alpha: Excitation matrix, shape (K, K)
                   alpha[i,j] = effect of event type j on type i
            beta: Decay rates, shape (K, K)
            T: Time horizon
            max_events: Maximum total events

        Returns:
            List of event lists, one per dimension
        """
        K = self.n_dimensions
        events = [[] for _ in range(K)]
        total_events = 0

        t = 0.0
        lambda_star = np.sum(mu) * 2  # Initial upper bound

        while t < T and total_events < max_events:
            # Compute current intensities
            intensities = self._compute_intensities(t, events, mu, alpha, beta)
            lambda_star = max(np.sum(intensities) * 1.5, 0.01)

            # Generate candidate event
            u = np.random.uniform()
            t = t - np.log(u) / lambda_star

            if t > T:
                break

            # Which process?
            intensities_t = self._compute_intensities(t, events, mu, alpha, beta)
            total_intensity = np.sum(intensities_t)

            D = np.random.uniform()
            if D * lambda_star <= total_intensity:
                # Accept event, determine which dimension
                probs = intensities_t / total_intensity
                dimension = np.random.choice(K, p=probs)
                events[dimension].append(t)
                total_events += 1

        return events

    def fit(self, events: List[List[float]], T: float) -> Dict:
        """
        Fit multivariate Hawkes process.

        Args:
            events: List of event lists, one per dimension
            T: Time horizon

        Returns:
            Dictionary with estimated parameters
        """
        K = self.n_dimensions

        # Convert to arrays
        events_arrays = [np.asarray(e) for e in events]

        # Initialize parameters
        n_events = [len(e) for e in events_arrays]
        mu_init = np.array([n / T for n in n_events])

        # Simple initialization: assume weak cross-excitation
        alpha_init = np.zeros((K, K))
        for i in range(K):
            alpha_init[i, i] = mu_init[i] * 0.3  # Self-excitation
            for j in range(K):
                if i != j:
                    alpha_init[i, j] = mu_init[i] * 0.1  # Cross-excitation

        beta_init = np.ones((K, K))

        # Flatten parameters for optimization
        def pack_params(mu, alpha, beta):
            return np.concatenate([mu.flatten(), alpha.flatten(), beta.flatten()])

        def unpack_params(x):
            mu = x[:K]
            alpha = x[K:K + K*K].reshape(K, K)
            beta = x[K + K*K:].reshape(K, K)
            return mu, alpha, beta

        # Negative log-likelihood
        def neg_log_likelihood(x):
            mu, alpha, beta = unpack_params(x)

            # Constraints
            if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
                return 1e10

            # Stability check (approximate)
            branching_ratios = alpha / beta
            if np.max(np.linalg.eigvals(branching_ratios).real) >= 0.99:
                return 1e10

            return -self._log_likelihood(events_arrays, T, mu, alpha, beta)

        # Optimize
        x0 = pack_params(mu_init, alpha_init, beta_init)
        bounds = [(1e-6, None)] * len(x0)  # All positive

        result = optimize.minimize(
            neg_log_likelihood,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        mu_opt, alpha_opt, beta_opt = unpack_params(result.x)
        log_likelihood = -result.fun

        # Information criteria
        n_params = len(x0)
        total_events = sum(n_events)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(total_events) * n_params

        return {
            'mu': mu_opt,
            'alpha': alpha_opt,
            'beta': beta_opt,
            'branching_matrix': alpha_opt / beta_opt,
            'spectral_radius': np.max(np.abs(np.linalg.eigvals(alpha_opt / beta_opt))),
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_events': n_events,
            'converged': result.success
        }

    def predict_intensities(self, events: List[List[float]],
                          mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                          t: float) -> np.ndarray:
        """
        Predict intensities for all dimensions at time t.

        Args:
            events: Past events
            mu, alpha, beta: Parameters
            t: Time to predict

        Returns:
            Intensity vector, shape (K,)
        """
        return self._compute_intensities(t, events, mu, alpha, beta)

    def _compute_intensities(self, t: float, events: List[List[float]],
                           mu: np.ndarray, alpha: np.ndarray,
                           beta: np.ndarray) -> np.ndarray:
        """Compute intensity vector at time t."""
        K = self.n_dimensions
        intensities = mu.copy()

        for k in range(K):
            for j in range(K):
                if len(events[j]) > 0:
                    events_j = np.asarray(events[j])
                    past_events = events_j[events_j < t]
                    if len(past_events) > 0:
                        excitation = alpha[k, j] * np.sum(
                            np.exp(-beta[k, j] * (t - past_events))
                        )
                        intensities[k] += excitation

        return intensities

    def _log_likelihood(self, events: List[np.ndarray], T: float,
                       mu: np.ndarray, alpha: np.ndarray,
                       beta: np.ndarray) -> float:
        """Compute log-likelihood for multivariate process."""
        K = self.n_dimensions
        log_sum = 0.0

        # First term: ∑_k ∑_i log(λ_k(t_i^k))
        for k in range(K):
            if len(events[k]) == 0:
                continue

            for i, t_i in enumerate(events[k]):
                # Need to compute λ_k(t_i) considering all past events
                events_up_to_i = [[] for _ in range(K)]
                for j in range(K):
                    events_up_to_i[j] = events[j][events[j] < t_i].tolist()

                intensities = self._compute_intensities(t_i, events_up_to_i, mu, alpha, beta)
                lambda_k = intensities[k]

                if lambda_k <= 0:
                    return -np.inf
                log_sum += np.log(lambda_k)

        # Second term: -∫_0^T ∑_k λ_k(s) ds
        integral = np.sum(mu) * T

        for k in range(K):
            for j in range(K):
                if len(events[j]) > 0:
                    integral += alpha[k, j] * np.sum(
                        (1 - np.exp(-beta[k, j] * (T - events[j]))) / beta[k, j]
                    )

        return log_sum - integral


class ConflictContagionModel:
    """
    Specialized Hawkes model for geopolitical conflict contagion.

    Features:
    - Models both self-excitation (conflict escalation within a country)
    - Models cross-excitation (conflict spreading between countries)
    - Incorporates spatial/network structure
    - Estimates contagion risk and early warning indicators

    Example:
        >>> countries = ['Syria', 'Iraq', 'Turkey']
        >>> model = ConflictContagionModel(countries=countries)
        >>>
        >>> # Fit to historical conflict events
        >>> events = {
        ...     'Syria': [1.2, 5.3, 10.1, ...],
        ...     'Iraq': [3.4, 8.9, ...],
        ...     'Turkey': [12.3, ...]
        ... }
        >>> result = model.fit(events, T=365.0)  # 1 year
        >>>
        >>> # Predict contagion risk
        >>> risk = model.contagion_risk(events, result, t=370.0)
        >>> print(f"Syria conflict risk in next 5 days: {risk['Syria']:.2%}")
    """

    def __init__(self, countries: List[str]):
        """
        Initialize conflict contagion model.

        Args:
            countries: List of country names
        """
        self.countries = countries
        self.n_countries = len(countries)
        self.hawkes = MultivariateHawkesProcess(n_dimensions=self.n_countries)

    def fit(self, events: Dict[str, List[float]], T: float) -> Dict:
        """
        Fit contagion model to conflict events.

        Args:
            events: Dictionary mapping country name to list of event times
            T: Observation period

        Returns:
            Fitted parameters with interpretation
        """
        # Convert to list format
        events_list = [events[country] for country in self.countries]

        # Fit multivariate Hawkes
        result = self.hawkes.fit(events_list, T)

        # Add interpretations
        result['countries'] = self.countries
        result['self_excitation'] = np.diag(result['alpha'])
        result['cross_excitation_mean'] = np.mean(
            result['alpha'][~np.eye(self.n_countries, dtype=bool)]
        )

        # Identify most contagious countries
        outgoing_contagion = np.sum(result['alpha'], axis=0) - np.diag(result['alpha'])
        incoming_contagion = np.sum(result['alpha'], axis=1) - np.diag(result['alpha'])

        result['most_contagious_source'] = self.countries[np.argmax(outgoing_contagion)]
        result['most_vulnerable_target'] = self.countries[np.argmax(incoming_contagion)]

        return result

    def contagion_risk(self, events: Dict[str, List[float]],
                      params: Dict, t: float, horizon: float = 5.0) -> Dict[str, float]:
        """
        Estimate contagion risk over next time period.

        Args:
            events: Historical events
            params: Fitted parameters
            t: Current time
            horizon: Risk horizon (time units)

        Returns:
            Dictionary mapping country to probability of conflict
        """
        events_list = [events[country] for country in self.countries]

        # Compute current intensities
        intensities = self.hawkes.predict_intensities(
            events_list, params['mu'], params['alpha'], params['beta'], t
        )

        # Probability of at least one event in [t, t+horizon]
        # P(N(t+h) - N(t) ≥ 1) = 1 - P(N(t+h) - N(t) = 0)
        # Approximate with constant intensity
        risks = {}
        for i, country in enumerate(self.countries):
            # Poisson approximation
            expected_events = intensities[i] * horizon
            prob_no_event = np.exp(-expected_events)
            prob_at_least_one = 1 - prob_no_event
            risks[country] = prob_at_least_one

        return risks

    def identify_contagion_pathways(self, params: Dict, threshold: float = 0.1) -> List[Tuple[str, str, float]]:
        """
        Identify significant contagion pathways between countries.

        Args:
            params: Fitted parameters
            threshold: Minimum branching ratio to report

        Returns:
            List of (source, target, branching_ratio) tuples
        """
        alpha = params['alpha']
        beta = params['beta']
        branching = alpha / beta

        pathways = []
        for i in range(self.n_countries):
            for j in range(self.n_countries):
                if i != j and branching[i, j] > threshold:
                    pathways.append((
                        self.countries[j],  # Source
                        self.countries[i],  # Target
                        branching[i, j]
                    ))

        # Sort by strength
        pathways.sort(key=lambda x: x[2], reverse=True)

        return pathways


def estimate_branching_ratio(events: np.ndarray, T: float) -> float:
    """
    Quick estimate of branching ratio for stability assessment.

    Args:
        events: Event times
        T: Time horizon

    Returns:
        Estimated branching ratio
    """
    hawkes = UnivariateHawkesProcess()
    result = hawkes.fit(events, T)
    return result.params.branching_ratio


def detect_explosive_regime(events: np.ndarray, T: float, window: float = 10.0) -> List[Tuple[float, float]]:
    """
    Detect time periods where process became explosive (supercritical).

    Args:
        events: Event times
        T: Total time horizon
        window: Rolling window size

    Returns:
        List of (start_time, branching_ratio) for explosive periods
    """
    events = np.sort(events)
    explosive_periods = []

    t = window
    while t <= T:
        # Events in window [t-window, t]
        window_events = events[(events >= t - window) & (events <= t)]

        if len(window_events) > 5:  # Need minimum events
            br = estimate_branching_ratio(window_events - (t - window), window)

            if br >= 0.9:  # Near or above critical
                explosive_periods.append((t, br))

        t += window / 2  # Overlapping windows

    return explosive_periods
