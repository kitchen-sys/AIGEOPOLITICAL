"""
Scenario representation and management for geopolitical modeling.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Scenario:
    """
    Represents a geopolitical scenario with multiple features and metadata.

    Attributes
    ----------
    name : str
        Name or identifier for the scenario
    features : Dict[str, np.ndarray]
        Dictionary of feature names to values
    timestamp : datetime
        Timestamp of the scenario
    metadata : Dict[str, Any]
        Additional metadata
    probability : float
        Probability or weight of this scenario (for ensembles)
    """
    name: str
    features: Dict[str, np.ndarray]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0

    def get_feature_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get features as a vector.

        Parameters
        ----------
        feature_names : List[str], optional
            List of feature names to include (if None, use all)

        Returns
        -------
        np.ndarray
            Feature vector
        """
        if feature_names is None:
            feature_names = list(self.features.keys())

        vectors = [self.features[name].flatten() for name in feature_names if name in self.features]
        return np.concatenate(vectors)

    def get_feature_matrix(self) -> np.ndarray:
        """
        Get all features as a matrix.

        Returns
        -------
        np.ndarray
            Feature matrix (n_features, ...)
        """
        return np.array([v for v in self.features.values()])

    def add_feature(self, name: str, values: np.ndarray) -> None:
        """
        Add a new feature to the scenario.

        Parameters
        ----------
        name : str
            Feature name
        values : np.ndarray
            Feature values
        """
        self.features[name] = values

    def remove_feature(self, name: str) -> None:
        """
        Remove a feature from the scenario.

        Parameters
        ----------
        name : str
            Feature name to remove
        """
        if name in self.features:
            del self.features[name]

    def clone(self) -> 'Scenario':
        """
        Create a deep copy of the scenario.

        Returns
        -------
        Scenario
            Cloned scenario
        """
        return Scenario(
            name=self.name,
            features={k: v.copy() for k, v in self.features.items()},
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
            probability=self.probability
        )


class ScenarioDistribution:
    """
    Represents a distribution over multiple scenarios.

    This is useful for Monte Carlo simulations, ensemble forecasting,
    and probabilistic reasoning.
    """

    def __init__(self, scenarios: Optional[List[Scenario]] = None):
        """
        Initialize scenario distribution.

        Parameters
        ----------
        scenarios : List[Scenario], optional
            Initial list of scenarios
        """
        self.scenarios: List[Scenario] = scenarios if scenarios is not None else []

    def add_scenario(self, scenario: Scenario) -> None:
        """
        Add a scenario to the distribution.

        Parameters
        ----------
        scenario : Scenario
            Scenario to add
        """
        self.scenarios.append(scenario)

    def get_probabilities(self) -> np.ndarray:
        """
        Get probabilities of all scenarios.

        Returns
        -------
        np.ndarray
            Array of probabilities
        """
        probs = np.array([s.probability for s in self.scenarios])
        # Normalize
        return probs / probs.sum()

    def normalize_probabilities(self) -> None:
        """
        Normalize scenario probabilities to sum to 1.
        """
        total_prob = sum(s.probability for s in self.scenarios)
        for scenario in self.scenarios:
            scenario.probability /= total_prob

    def get_feature_samples(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get feature samples from all scenarios.

        Parameters
        ----------
        feature_names : List[str], optional
            List of feature names to include

        Returns
        -------
        np.ndarray
            Feature samples (n_scenarios, n_features)
        """
        samples = [s.get_feature_vector(feature_names) for s in self.scenarios]
        return np.array(samples)

    def get_weighted_mean(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute weighted mean of features.

        Parameters
        ----------
        feature_names : List[str], optional
            List of feature names to include

        Returns
        -------
        np.ndarray
            Weighted mean feature vector
        """
        samples = self.get_feature_samples(feature_names)
        probs = self.get_probabilities()
        return np.average(samples, axis=0, weights=probs)

    def get_variance(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute variance of features.

        Parameters
        ----------
        feature_names : List[str], optional
            List of feature names to include

        Returns
        -------
        np.ndarray
            Variance of features
        """
        samples = self.get_feature_samples(feature_names)
        probs = self.get_probabilities()
        mean = self.get_weighted_mean(feature_names)

        variance = np.average((samples - mean) ** 2, axis=0, weights=probs)
        return variance

    def sample(self, n_samples: int = 1, replace: bool = True) -> List[Scenario]:
        """
        Sample scenarios from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        replace : bool
            Whether to sample with replacement

        Returns
        -------
        List[Scenario]
            Sampled scenarios
        """
        probs = self.get_probabilities()
        indices = np.random.choice(
            len(self.scenarios),
            size=n_samples,
            replace=replace,
            p=probs
        )
        return [self.scenarios[i] for i in indices]

    def filter_by_probability(self, threshold: float) -> 'ScenarioDistribution':
        """
        Filter scenarios by probability threshold.

        Parameters
        ----------
        threshold : float
            Minimum probability threshold

        Returns
        -------
        ScenarioDistribution
            New distribution with filtered scenarios
        """
        filtered_scenarios = [s for s in self.scenarios if s.probability >= threshold]
        return ScenarioDistribution(filtered_scenarios)

    def get_top_k(self, k: int) -> 'ScenarioDistribution':
        """
        Get top k scenarios by probability.

        Parameters
        ----------
        k : int
            Number of scenarios to return

        Returns
        -------
        ScenarioDistribution
            Distribution with top k scenarios
        """
        sorted_scenarios = sorted(self.scenarios, key=lambda s: s.probability, reverse=True)
        return ScenarioDistribution(sorted_scenarios[:k])

    def __len__(self) -> int:
        """Return number of scenarios."""
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Scenario:
        """Get scenario by index."""
        return self.scenarios[idx]

    def __iter__(self):
        """Iterate over scenarios."""
        return iter(self.scenarios)
