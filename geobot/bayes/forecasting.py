"""
Bayesian Forecasting Module for GeoBotv1

Provides Bayesian belief updating, prior construction, and probabilistic forecasting
for geopolitical scenarios. Integrates with GeoBot 2.0 analytical framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
import numpy as np
from scipy import stats
from scipy.optimize import minimize


class PriorType(Enum):
    """Types of prior distributions."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    BETA = "beta"
    GAMMA = "gamma"
    EXPERT_INFORMED = "expert_informed"
    HISTORICAL = "historical"


class EvidenceType(Enum):
    """Types of evidence for belief updating."""
    INTELLIGENCE_REPORT = "intelligence_report"
    SATELLITE_IMAGERY = "satellite_imagery"
    ECONOMIC_DATA = "economic_data"
    MILITARY_MOVEMENT = "military_movement"
    DIPLOMATIC_SIGNAL = "diplomatic_signal"
    OPEN_SOURCE = "open_source"


@dataclass
class GeopoliticalPrior:
    """
    Prior distribution for geopolitical parameter.

    Attributes
    ----------
    parameter_name : str
        Name of the parameter
    prior_type : PriorType
        Type of prior distribution
    parameters : Dict[str, float]
        Distribution parameters
    description : str
        Description of what this parameter represents
    """
    parameter_name: str
    prior_type: PriorType
    parameters: Dict[str, float]
    description: str = ""

    def sample(self, n_samples: int = 1000, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from prior distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples
        random_state : Optional[int]
            Random seed

        Returns
        -------
        np.ndarray
            Samples from prior
        """
        if random_state is not None:
            np.random.seed(random_state)

        if self.prior_type == PriorType.UNIFORM:
            low = self.parameters['low']
            high = self.parameters['high']
            return np.random.uniform(low, high, n_samples)

        elif self.prior_type == PriorType.NORMAL:
            mean = self.parameters['mean']
            std = self.parameters['std']
            return np.random.normal(mean, std, n_samples)

        elif self.prior_type == PriorType.BETA:
            alpha = self.parameters['alpha']
            beta = self.parameters['beta']
            return np.random.beta(alpha, beta, n_samples)

        elif self.prior_type == PriorType.GAMMA:
            shape = self.parameters['shape']
            scale = self.parameters['scale']
            return np.random.gamma(shape, scale, n_samples)

        else:
            raise ValueError(f"Sampling not implemented for {self.prior_type}")

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute probability density function.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate PDF

        Returns
        -------
        np.ndarray
            PDF values
        """
        if self.prior_type == PriorType.UNIFORM:
            low = self.parameters['low']
            high = self.parameters['high']
            return stats.uniform.pdf(x, loc=low, scale=high-low)

        elif self.prior_type == PriorType.NORMAL:
            mean = self.parameters['mean']
            std = self.parameters['std']
            return stats.norm.pdf(x, loc=mean, scale=std)

        elif self.prior_type == PriorType.BETA:
            alpha = self.parameters['alpha']
            beta = self.parameters['beta']
            return stats.beta.pdf(x, alpha, beta)

        elif self.prior_type == PriorType.GAMMA:
            shape = self.parameters['shape']
            scale = self.parameters['scale']
            return stats.gamma.pdf(x, shape, scale=scale)

        else:
            raise ValueError(f"PDF not implemented for {self.prior_type}")


@dataclass
class EvidenceUpdate:
    """
    Evidence for Bayesian belief updating.

    Attributes
    ----------
    evidence_type : EvidenceType
        Type of evidence
    observation : Any
        Observed value or data
    likelihood_function : Callable
        Function mapping parameters to likelihood of observation
    reliability : float
        Reliability score [0, 1]
    source : str
        Source of evidence
    timestamp : Optional[str]
        When evidence was collected
    """
    evidence_type: EvidenceType
    observation: Any
    likelihood_function: Callable[[np.ndarray], np.ndarray]
    reliability: float = 1.0
    source: str = ""
    timestamp: Optional[str] = None

    def compute_likelihood(self, parameter_values: np.ndarray) -> np.ndarray:
        """
        Compute likelihood of observation given parameter values.

        Parameters
        ----------
        parameter_values : np.ndarray
            Parameter values

        Returns
        -------
        np.ndarray
            Likelihood values
        """
        base_likelihood = self.likelihood_function(parameter_values)
        # Adjust for reliability
        return base_likelihood ** self.reliability


@dataclass
class BeliefState:
    """
    Current belief state (posterior distribution).

    Attributes
    ----------
    parameter_name : str
        Name of parameter
    posterior_samples : np.ndarray
        Samples from posterior distribution
    prior : GeopoliticalPrior
        Original prior
    evidence_history : List[EvidenceUpdate]
        Evidence used to update beliefs
    """
    parameter_name: str
    posterior_samples: np.ndarray
    prior: GeopoliticalPrior
    evidence_history: List[EvidenceUpdate] = field(default_factory=list)

    def mean(self) -> float:
        """Posterior mean."""
        return float(np.mean(self.posterior_samples))

    def median(self) -> float:
        """Posterior median."""
        return float(np.median(self.posterior_samples))

    def std(self) -> float:
        """Posterior standard deviation."""
        return float(np.std(self.posterior_samples))

    def quantile(self, q: float) -> float:
        """
        Posterior quantile.

        Parameters
        ----------
        q : float
            Quantile in [0, 1]

        Returns
        -------
        float
            Quantile value
        """
        return float(np.quantile(self.posterior_samples, q))

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Compute credible interval.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI)

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds
        """
        lower = self.quantile(alpha / 2)
        upper = self.quantile(1 - alpha / 2)
        return (lower, upper)

    def probability_greater_than(self, threshold: float) -> float:
        """
        Compute P(parameter > threshold | evidence).

        Parameters
        ----------
        threshold : float
            Threshold value

        Returns
        -------
        float
            Probability
        """
        return float(np.mean(self.posterior_samples > threshold))

    def probability_in_range(self, low: float, high: float) -> float:
        """
        Compute P(low < parameter < high | evidence).

        Parameters
        ----------
        low : float
            Lower bound
        high : float
            Upper bound

        Returns
        -------
        float
            Probability
        """
        return float(np.mean((self.posterior_samples > low) &
                           (self.posterior_samples < high)))


@dataclass
class CredibleInterval:
    """Credible interval for forecast."""
    lower: float
    upper: float
    alpha: float  # Significance level

    @property
    def width(self) -> float:
        """Interval width."""
        return self.upper - self.lower

    @property
    def credibility(self) -> float:
        """Credibility level (e.g., 0.95 for 95% CI)."""
        return 1 - self.alpha


@dataclass
class ForecastDistribution:
    """
    Predictive distribution for geopolitical forecast.

    Attributes
    ----------
    variable_name : str
        Name of forecasted variable
    samples : np.ndarray
        Samples from predictive distribution
    time_horizon : int
        Forecast horizon (days, months, etc.)
    conditioning_info : Dict[str, Any]
        Information conditioned on
    """
    variable_name: str
    samples: np.ndarray
    time_horizon: int
    conditioning_info: Dict[str, Any] = field(default_factory=dict)

    def point_forecast(self, method: str = 'mean') -> float:
        """
        Point forecast.

        Parameters
        ----------
        method : str
            'mean', 'median', or 'mode'

        Returns
        -------
        float
            Point forecast
        """
        if method == 'mean':
            return float(np.mean(self.samples))
        elif method == 'median':
            return float(np.median(self.samples))
        elif method == 'mode':
            # Use kernel density estimation for mode
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(self.samples)
            x = np.linspace(self.samples.min(), self.samples.max(), 1000)
            return float(x[np.argmax(kde(x))])
        else:
            raise ValueError(f"Unknown method: {method}")

    def credible_interval(self, alpha: float = 0.05) -> CredibleInterval:
        """
        Compute credible interval.

        Parameters
        ----------
        alpha : float
            Significance level

        Returns
        -------
        CredibleInterval
            Credible interval
        """
        lower = float(np.quantile(self.samples, alpha / 2))
        upper = float(np.quantile(self.samples, 1 - alpha / 2))
        return CredibleInterval(lower=lower, upper=upper, alpha=alpha)

    def probability_of_event(self, condition: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Probability of event defined by condition.

        Parameters
        ----------
        condition : Callable
            Function that returns True/False for each sample

        Returns
        -------
        float
            Probability
        """
        return float(np.mean(condition(self.samples)))


class BayesianForecaster:
    """
    Bayesian forecasting engine for geopolitical analysis.

    Integrates with GeoBot 2.0 analytical framework to provide
    probabilistic forecasts with explicit uncertainty quantification.
    """

    def __init__(self):
        """Initialize Bayesian forecaster."""
        self.priors: Dict[str, GeopoliticalPrior] = {}
        self.beliefs: Dict[str, BeliefState] = {}

    def set_prior(self, prior: GeopoliticalPrior) -> None:
        """
        Set prior distribution for parameter.

        Parameters
        ----------
        prior : GeopoliticalPrior
            Prior distribution
        """
        self.priors[prior.parameter_name] = prior

    def update_belief(
        self,
        parameter_name: str,
        evidence: EvidenceUpdate,
        n_samples: int = 10000,
        method: str = 'importance_sampling'
    ) -> BeliefState:
        """
        Update beliefs using Bayes' rule.

        Parameters
        ----------
        parameter_name : str
            Parameter to update
        evidence : EvidenceUpdate
            New evidence
        n_samples : int
            Number of samples for approximation
        method : str
            'importance_sampling' or 'rejection_sampling'

        Returns
        -------
        BeliefState
            Updated belief state
        """
        if parameter_name not in self.priors:
            raise ValueError(f"No prior set for {parameter_name}")

        # Get current prior or posterior
        if parameter_name in self.beliefs:
            # Use previous posterior as new prior
            prior_samples = self.beliefs[parameter_name].posterior_samples
        else:
            # Use original prior
            prior_samples = self.priors[parameter_name].sample(n_samples)

        # Compute likelihoods
        likelihoods = evidence.compute_likelihood(prior_samples)

        if method == 'importance_sampling':
            # Importance sampling with resampling
            weights = likelihoods / np.sum(likelihoods)

            # Resample according to weights
            indices = np.random.choice(
                len(prior_samples),
                size=n_samples,
                replace=True,
                p=weights
            )
            posterior_samples = prior_samples[indices]

        elif method == 'rejection_sampling':
            # Rejection sampling
            max_likelihood = np.max(likelihoods)
            accepted = []

            for sample, likelihood in zip(prior_samples, likelihoods):
                if np.random.uniform(0, max_likelihood) < likelihood:
                    accepted.append(sample)

            if len(accepted) < 100:
                raise ValueError("Rejection sampling failed - too few accepted samples")

            posterior_samples = np.array(accepted)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Create or update belief state
        if parameter_name in self.beliefs:
            belief = self.beliefs[parameter_name]
            belief.posterior_samples = posterior_samples
            belief.evidence_history.append(evidence)
        else:
            belief = BeliefState(
                parameter_name=parameter_name,
                posterior_samples=posterior_samples,
                prior=self.priors[parameter_name],
                evidence_history=[evidence]
            )

        self.beliefs[parameter_name] = belief
        return belief

    def sequential_update(
        self,
        parameter_name: str,
        evidence_sequence: List[EvidenceUpdate],
        n_samples: int = 10000
    ) -> BeliefState:
        """
        Sequential belief updating with multiple pieces of evidence.

        Parameters
        ----------
        parameter_name : str
            Parameter to update
        evidence_sequence : List[EvidenceUpdate]
            Sequence of evidence
        n_samples : int
            Number of samples

        Returns
        -------
        BeliefState
            Final belief state
        """
        for evidence in evidence_sequence:
            self.update_belief(parameter_name, evidence, n_samples)

        return self.beliefs[parameter_name]

    def forecast(
        self,
        variable_name: str,
        predictive_function: Callable[[Dict[str, float]], float],
        time_horizon: int,
        n_samples: int = 10000,
        conditioning_info: Optional[Dict[str, Any]] = None
    ) -> ForecastDistribution:
        """
        Generate probabilistic forecast.

        Parameters
        ----------
        variable_name : str
            Variable to forecast
        predictive_function : Callable
            Function mapping parameter values to prediction
        time_horizon : int
            Forecast horizon
        n_samples : int
            Number of forecast samples
        conditioning_info : Optional[Dict[str, Any]]
            Additional conditioning information

        Returns
        -------
        ForecastDistribution
            Forecast distribution
        """
        # Sample parameters from beliefs
        parameter_samples = {}
        for param_name, belief in self.beliefs.items():
            indices = np.random.choice(len(belief.posterior_samples), size=n_samples)
            parameter_samples[param_name] = belief.posterior_samples[indices]

        # Generate forecasts
        forecast_samples = np.zeros(n_samples)
        for i in range(n_samples):
            params = {name: samples[i] for name, samples in parameter_samples.items()}
            forecast_samples[i] = predictive_function(params)

        return ForecastDistribution(
            variable_name=variable_name,
            samples=forecast_samples,
            time_horizon=time_horizon,
            conditioning_info=conditioning_info or {}
        )

    def model_comparison(
        self,
        models: Dict[str, Callable],
        evidence: List[EvidenceUpdate],
        prior_model_probs: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Bayesian model comparison using evidence.

        Parameters
        ----------
        models : Dict[str, Callable]
            Dictionary of models (name -> likelihood function)
        evidence : List[EvidenceUpdate]
            Evidence for comparison
        prior_model_probs : Optional[Dict[str, float]]
            Prior model probabilities

        Returns
        -------
        Dict[str, float]
            Posterior model probabilities
        """
        if prior_model_probs is None:
            # Uniform prior over models
            prior_model_probs = {name: 1.0 / len(models) for name in models}

        # Compute marginal likelihoods (evidence)
        marginal_likelihoods = {}

        for model_name, model_fn in models.items():
            # This is a simplified version - full implementation would
            # integrate over parameter space
            likelihood = 1.0
            for ev in evidence:
                # Assuming model_fn can compute likelihood
                likelihood *= np.mean(ev.compute_likelihood(model_fn))

            marginal_likelihoods[model_name] = likelihood

        # Compute posterior model probabilities
        posterior_probs = {}
        total = 0.0

        for model_name in models:
            unnormalized = (prior_model_probs[model_name] *
                          marginal_likelihoods[model_name])
            posterior_probs[model_name] = unnormalized
            total += unnormalized

        # Normalize
        for model_name in posterior_probs:
            posterior_probs[model_name] /= total

        return posterior_probs

    def get_belief_summary(self, parameter_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for belief state.

        Parameters
        ----------
        parameter_name : str
            Parameter name

        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if parameter_name not in self.beliefs:
            raise ValueError(f"No beliefs for {parameter_name}")

        belief = self.beliefs[parameter_name]
        ci_95 = belief.credible_interval(alpha=0.05)
        ci_90 = belief.credible_interval(alpha=0.10)

        return {
            'parameter': parameter_name,
            'mean': belief.mean(),
            'median': belief.median(),
            'std': belief.std(),
            '95%_CI': ci_95,
            '90%_CI': ci_90,
            'n_evidence_updates': len(belief.evidence_history),
            'evidence_types': [ev.evidence_type.value for ev in belief.evidence_history]
        }
