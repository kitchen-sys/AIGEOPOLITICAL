"""
Bayesian Inference Engine

Provides principled way to update beliefs as new intelligence, rumors,
events, or data arrive.

Components:
- Priors: Baseline beliefs
- Likelihood: Evidence
- Posteriors: Updated beliefs

Necessary for:
- Real-time updates
- Intelligence feeds
- Event-driven recalibration
- Uncertainty tracking

Monte Carlo + Bayesian updating = elite forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class Prior:
    """
    Represents a prior distribution.

    Attributes
    ----------
    name : str
        Name of the variable
    distribution : Any
        Prior distribution (scipy.stats distribution)
    parameters : dict
        Distribution parameters
    """
    name: str
    distribution: Any
    parameters: Dict[str, float]

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from prior.

        Parameters
        ----------
        n_samples : int
            Number of samples

        Returns
        -------
        np.ndarray
            Samples from prior
        """
        return self.distribution.rvs(size=n_samples, **self.parameters)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate prior probability density.

        Parameters
        ----------
        x : np.ndarray
            Points to evaluate

        Returns
        -------
        np.ndarray
            Probability densities
        """
        return self.distribution.pdf(x, **self.parameters)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate log prior probability density.

        Parameters
        ----------
        x : np.ndarray
            Points to evaluate

        Returns
        -------
        np.ndarray
            Log probability densities
        """
        return self.distribution.logpdf(x, **self.parameters)


@dataclass
class Evidence:
    """
    Represents evidence/observation.

    Attributes
    ----------
    observation : Any
        Observed data
    likelihood_fn : Callable
        Likelihood function
    timestamp : float
        Time of observation
    confidence : float
        Confidence in observation (0-1)
    """
    observation: Any
    likelihood_fn: Callable
    timestamp: float
    confidence: float = 1.0


class BayesianEngine:
    """
    Bayesian inference engine for belief updating.

    This engine maintains and updates probability distributions
    as new evidence arrives, enabling real-time forecasting with
    uncertainty quantification.
    """

    def __init__(self):
        """Initialize Bayesian engine."""
        self.priors: Dict[str, Prior] = {}
        self.posteriors: Dict[str, np.ndarray] = {}
        self.evidence_history: List[Evidence] = []

    def set_prior(self, prior: Prior) -> None:
        """
        Set prior distribution for a variable.

        Parameters
        ----------
        prior : Prior
            Prior distribution
        """
        self.priors[prior.name] = prior

    def update(
        self,
        variable: str,
        evidence: Evidence,
        method: str = 'grid'
    ) -> np.ndarray:
        """
        Update beliefs given evidence.

        Parameters
        ----------
        variable : str
            Variable to update
        evidence : Evidence
            New evidence
        method : str
            Update method ('grid', 'mcmc', 'analytical')

        Returns
        -------
        np.ndarray
            Posterior samples
        """
        if variable not in self.priors:
            raise ValueError(f"No prior set for {variable}")

        self.evidence_history.append(evidence)

        if method == 'grid':
            posterior = self._grid_update(variable, evidence)
        elif method == 'mcmc':
            posterior = self._mcmc_update(variable, evidence)
        elif method == 'analytical':
            posterior = self._analytical_update(variable, evidence)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.posteriors[variable] = posterior
        return posterior

    def _grid_update(self, variable: str, evidence: Evidence) -> np.ndarray:
        """
        Grid approximation for Bayesian update.

        Parameters
        ----------
        variable : str
            Variable name
        evidence : Evidence
            Evidence

        Returns
        -------
        np.ndarray
            Posterior samples
        """
        prior = self.priors[variable]

        # Create grid based on distribution type
        n_grid = 1000

        # Check distribution type by name
        dist_name = prior.distribution.name if hasattr(prior.distribution, 'name') else 'unknown'

        if dist_name == 'beta':
            # Beta distribution always has support [0, 1]
            grid = np.linspace(0, 1, n_grid)
        elif dist_name == 'gamma':
            # Gamma distribution has support [0, inf)
            # Use reasonable upper bound based on parameters
            shape = prior.parameters.get('a', 1)
            scale = prior.parameters.get('scale', 1)
            mean = shape * scale
            std = np.sqrt(shape) * scale
            grid = np.linspace(0, mean + 4*std, n_grid)
        elif dist_name == 'uniform':
            # Uniform distribution uses loc and scale
            loc = prior.parameters.get('loc', 0)
            scale = prior.parameters.get('scale', 1)
            grid = np.linspace(loc, loc + scale, n_grid)
        else:
            # Default for normal and other distributions
            mean = prior.parameters.get('loc', 0)
            std = prior.parameters.get('scale', 1)
            grid = np.linspace(mean - 4*std, mean + 4*std, n_grid)

        # Compute prior * likelihood
        prior_vals = prior.pdf(grid)
        likelihood_vals = evidence.likelihood_fn(grid, evidence.observation)

        # Weight by evidence confidence
        likelihood_vals = likelihood_vals ** evidence.confidence

        # Compute posterior (unnormalized)
        posterior_vals = prior_vals * likelihood_vals

        # Normalize
        posterior_vals /= posterior_vals.sum()

        # Sample from posterior
        n_samples = 10000
        posterior_samples = np.random.choice(grid, size=n_samples, p=posterior_vals)

        return posterior_samples

    def _mcmc_update(
        self,
        variable: str,
        evidence: Evidence,
        n_samples: int = 10000
    ) -> np.ndarray:
        """
        MCMC-based Bayesian update.

        Parameters
        ----------
        variable : str
            Variable name
        evidence : Evidence
            Evidence
        n_samples : int
            Number of MCMC samples

        Returns
        -------
        np.ndarray
            Posterior samples
        """
        prior = self.priors[variable]

        def log_posterior(x):
            log_prior = prior.log_pdf(np.array([x]))[0]
            log_likelihood = np.log(evidence.likelihood_fn(np.array([x]), evidence.observation)[0] + 1e-10)
            return log_prior + evidence.confidence * log_likelihood

        # Simple Metropolis-Hastings
        samples = []
        current = prior.sample(1)[0]
        current_log_p = log_posterior(current)

        for _ in range(n_samples):
            # Propose
            proposal = current + np.random.normal(0, 0.1)
            proposal_log_p = log_posterior(proposal)

            # Accept/reject
            log_alpha = proposal_log_p - current_log_p
            if np.log(np.random.uniform()) < log_alpha:
                current = proposal
                current_log_p = proposal_log_p

            samples.append(current)

        return np.array(samples)

    def _analytical_update(self, variable: str, evidence: Evidence) -> np.ndarray:
        """
        Analytical Bayesian update (for conjugate priors).

        Parameters
        ----------
        variable : str
            Variable name
        evidence : Evidence
            Evidence

        Returns
        -------
        np.ndarray
            Posterior samples
        """
        # Placeholder - would implement conjugate updates
        # For now, fall back to grid
        return self._grid_update(variable, evidence)

    def get_posterior_summary(self, variable: str) -> Dict[str, float]:
        """
        Get summary statistics of posterior.

        Parameters
        ----------
        variable : str
            Variable name

        Returns
        -------
        dict
            Summary statistics
        """
        if variable not in self.posteriors:
            raise ValueError(f"No posterior for {variable}")

        samples = self.posteriors[variable]

        return {
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'q5': np.percentile(samples, 5),
            'q25': np.percentile(samples, 25),
            'q75': np.percentile(samples, 75),
            'q95': np.percentile(samples, 95)
        }

    def get_credible_interval(
        self,
        variable: str,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Get credible interval for posterior.

        Parameters
        ----------
        variable : str
            Variable name
        alpha : float
            Significance level

        Returns
        -------
        tuple
            (lower, upper) bounds of credible interval
        """
        if variable not in self.posteriors:
            raise ValueError(f"No posterior for {variable}")

        samples = self.posteriors[variable]
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))

        return lower, upper

    def compute_bayes_factor(
        self,
        variable: str,
        hypothesis1: Callable,
        hypothesis2: Callable
    ) -> float:
        """
        Compute Bayes factor for two hypotheses.

        Parameters
        ----------
        variable : str
            Variable name
        hypothesis1 : callable
            First hypothesis (returns bool)
        hypothesis2 : callable
            Second hypothesis (returns bool)

        Returns
        -------
        float
            Bayes factor (BF > 1 favors hypothesis1)
        """
        if variable not in self.posteriors:
            raise ValueError(f"No posterior for {variable}")

        samples = self.posteriors[variable]

        p1 = np.mean([hypothesis1(x) for x in samples])
        p2 = np.mean([hypothesis2(x) for x in samples])

        if p2 == 0:
            return np.inf
        return p1 / p2


class BeliefUpdater:
    """
    High-level interface for updating geopolitical beliefs.

    This class provides domain-specific methods for updating
    beliefs based on intelligence, events, and rumors.
    """

    def __init__(self):
        """Initialize belief updater."""
        self.engine = BayesianEngine()
        self.beliefs: Dict[str, Dict[str, Any]] = {}

    def initialize_belief(
        self,
        name: str,
        prior_mean: float,
        prior_std: float,
        belief_type: str = 'continuous'
    ) -> None:
        """
        Initialize a belief with prior.

        Parameters
        ----------
        name : str
            Belief name
        prior_mean : float
            Prior mean
        prior_std : float
            Prior standard deviation
        belief_type : str
            Type of belief ('continuous', 'probability')
        """
        if belief_type == 'continuous':
            distribution = stats.norm
            parameters = {'loc': prior_mean, 'scale': prior_std}
        elif belief_type == 'probability':
            # Use beta distribution for probabilities
            # Convert mean/std to alpha/beta parameters
            mean = np.clip(prior_mean, 0.01, 0.99)
            var = prior_std ** 2
            alpha = mean * (mean * (1 - mean) / var - 1)
            beta = (1 - mean) * (mean * (1 - mean) / var - 1)
            distribution = stats.beta
            # Beta uses 'a' and 'b' parameters, not 'loc' and 'scale'
            parameters = {'a': alpha, 'b': beta}
        else:
            distribution = stats.norm
            parameters = {'loc': prior_mean, 'scale': prior_std}

        prior = Prior(
            name=name,
            distribution=distribution,
            parameters=parameters
        )

        self.engine.set_prior(prior)
        self.beliefs[name] = {
            'type': belief_type,
            'initialized': True
        }

    def update_from_intelligence(
        self,
        belief: str,
        observation: float,
        reliability: float = 0.8
    ) -> Dict[str, float]:
        """
        Update belief from intelligence report.

        Parameters
        ----------
        belief : str
            Belief name
        observation : float
            Observed value from intelligence
        reliability : float
            Reliability of intelligence source (0-1)

        Returns
        -------
        dict
            Posterior summary
        """
        def likelihood_fn(x, obs):
            # Gaussian likelihood centered at observation
            # Width depends on reliability
            std = 1.0 / reliability
            return stats.norm.pdf(x, loc=obs, scale=std)

        evidence = Evidence(
            observation=observation,
            likelihood_fn=likelihood_fn,
            timestamp=pd.Timestamp.now().timestamp(),
            confidence=reliability
        )

        self.engine.update(belief, evidence, method='grid')
        return self.engine.get_posterior_summary(belief)

    def update_from_event(
        self,
        belief: str,
        event_impact: float,
        event_certainty: float = 1.0
    ) -> Dict[str, float]:
        """
        Update belief from observed event.

        Parameters
        ----------
        belief : str
            Belief name
        event_impact : float
            Impact of event (shift in belief)
        event_certainty : float
            Certainty that event occurred (0-1)

        Returns
        -------
        dict
            Posterior summary
        """
        # Get current belief
        if belief not in self.engine.posteriors:
            # Use prior
            current_samples = self.engine.priors[belief].sample(10000)
        else:
            current_samples = self.engine.posteriors[belief]

        current_mean = np.mean(current_samples)

        # Create shifted observation
        observation = current_mean + event_impact

        def likelihood_fn(x, obs):
            return stats.norm.pdf(x, loc=obs, scale=abs(event_impact) * 0.1)

        evidence = Evidence(
            observation=observation,
            likelihood_fn=likelihood_fn,
            timestamp=pd.Timestamp.now().timestamp(),
            confidence=event_certainty
        )

        self.engine.update(belief, evidence, method='grid')
        return self.engine.get_posterior_summary(belief)

    def get_belief_probability(
        self,
        belief: str,
        threshold: float,
        direction: str = 'greater'
    ) -> float:
        """
        Get probability that belief exceeds threshold.

        Parameters
        ----------
        belief : str
            Belief name
        threshold : float
            Threshold value
        direction : str
            'greater' or 'less'

        Returns
        -------
        float
            Probability
        """
        if belief not in self.engine.posteriors:
            samples = self.engine.priors[belief].sample(10000)
        else:
            samples = self.engine.posteriors[belief]

        if direction == 'greater':
            return np.mean(samples > threshold)
        else:
            return np.mean(samples < threshold)

    def compare_beliefs(self, belief1: str, belief2: str) -> Dict[str, float]:
        """
        Compare two beliefs.

        Parameters
        ----------
        belief1 : str
            First belief
        belief2 : str
            Second belief

        Returns
        -------
        dict
            Comparison results
        """
        if belief1 not in self.engine.posteriors:
            samples1 = self.engine.priors[belief1].sample(10000)
        else:
            samples1 = self.engine.posteriors[belief1]

        if belief2 not in self.engine.posteriors:
            samples2 = self.engine.priors[belief2].sample(10000)
        else:
            samples2 = self.engine.posteriors[belief2]

        return {
            'p_belief1_greater': np.mean(samples1 > samples2),
            'mean_difference': np.mean(samples1 - samples2),
            'correlation': np.corrcoef(samples1, samples2)[0, 1]
        }
