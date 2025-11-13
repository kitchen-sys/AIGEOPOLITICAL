"""
Optimal Transport Module - Wasserstein Distances

Provides geometric measures of how much "effort" is needed to move from
one geopolitical scenario to another using optimal transport theory.

Applications:
- Measure regime shifts
- Compare distributions of Monte Carlo futures
- Quantify shock impact
- Measure closeness of geopolitical scenarios
- Detect structural change
- Logistics modeling
"""

import numpy as np
from typing import Union, Tuple, Optional, List
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

try:
    import ot  # Python Optimal Transport library
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("Warning: POT library not available. Some features will be limited.")


class WassersteinDistance:
    """
    Compute Wasserstein distances between probability distributions.

    The Wasserstein distance (also known as Earth Mover's Distance) provides
    a principled way to measure the distance between probability distributions,
    accounting for the geometry of the underlying space.
    """

    def __init__(self, metric: str = 'euclidean', p: int = 2):
        """
        Initialize Wasserstein distance calculator.

        Parameters
        ----------
        metric : str
            Distance metric to use for ground distance ('euclidean', 'cityblock', etc.)
        p : int
            Order of Wasserstein distance (1 or 2)
        """
        self.metric = metric
        self.p = p

    def compute_1d(
        self,
        u_values: np.ndarray,
        v_values: np.ndarray,
        u_weights: Optional[np.ndarray] = None,
        v_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute 1D Wasserstein distance between two distributions.

        Parameters
        ----------
        u_values : np.ndarray
            Values for first distribution
        v_values : np.ndarray
            Values for second distribution
        u_weights : np.ndarray, optional
            Weights for first distribution (defaults to uniform)
        v_weights : np.ndarray, optional
            Weights for second distribution (defaults to uniform)

        Returns
        -------
        float
            Wasserstein distance
        """
        return wasserstein_distance(u_values, v_values, u_weights, v_weights)

    def compute_nd(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        method: str = 'sinkhorn'
    ) -> float:
        """
        Compute n-dimensional Wasserstein distance.

        Parameters
        ----------
        X_source : np.ndarray, shape (n_samples_source, n_features)
            Source distribution samples
        X_target : np.ndarray, shape (n_samples_target, n_features)
            Target distribution samples
        a : np.ndarray, optional
            Weights for source distribution (defaults to uniform)
        b : np.ndarray, optional
            Weights for target distribution (defaults to uniform)
        method : str
            Method to use ('sinkhorn', 'emd', 'emd2')

        Returns
        -------
        float
            Wasserstein distance
        """
        if not HAS_POT:
            raise ImportError("POT library required for n-dimensional distances")

        n_source = X_source.shape[0]
        n_target = X_target.shape[0]

        # Default to uniform distributions
        if a is None:
            a = np.ones(n_source) / n_source
        if b is None:
            b = np.ones(n_target) / n_target

        # Compute cost matrix
        M = cdist(X_source, X_target, metric=self.metric)

        # Compute optimal transport
        if method == 'sinkhorn':
            # Sinkhorn algorithm (faster, approximate)
            distance = ot.sinkhorn2(a, b, M, reg=0.1)
        elif method == 'emd':
            # Exact EMD
            distance = ot.emd2(a, b, M)
        elif method == 'emd2':
            # Squared EMD
            distance = ot.emd2(a, b, M**2)
        else:
            raise ValueError(f"Unknown method: {method}")

        return float(distance)

    def compute_barycenter(
        self,
        distributions: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
        method: str = 'sinkhorn'
    ) -> np.ndarray:
        """
        Compute Wasserstein barycenter of multiple distributions.

        This finds the "average" distribution in Wasserstein space.

        Parameters
        ----------
        distributions : list of np.ndarray
            List of distributions to average
        weights : np.ndarray, optional
            Weights for each distribution
        method : str
            Method to use ('sinkhorn')

        Returns
        -------
        np.ndarray
            Wasserstein barycenter
        """
        if not HAS_POT:
            raise ImportError("POT library required for barycenter computation")

        n_distributions = len(distributions)

        if weights is None:
            weights = np.ones(n_distributions) / n_distributions

        # Stack distributions
        A = np.column_stack(distributions)

        # Compute barycenter
        if method == 'sinkhorn':
            barycenter = ot.bregman.barycenter(A, M=None, reg=0.1, weights=weights)
        else:
            raise ValueError(f"Unknown method: {method}")

        return barycenter


class ScenarioComparator:
    """
    Compare geopolitical scenarios using optimal transport.

    This class provides high-level methods for comparing scenarios,
    detecting regime shifts, and quantifying shock impacts.
    """

    def __init__(self, metric: str = 'euclidean'):
        """
        Initialize scenario comparator.

        Parameters
        ----------
        metric : str
            Distance metric for ground distance
        """
        self.wasserstein = WassersteinDistance(metric=metric)

    def compare_scenarios(
        self,
        scenario1: np.ndarray,
        scenario2: np.ndarray,
        weights1: Optional[np.ndarray] = None,
        weights2: Optional[np.ndarray] = None
    ) -> float:
        """
        Compare two geopolitical scenarios.

        Parameters
        ----------
        scenario1 : np.ndarray
            First scenario (features x samples)
        scenario2 : np.ndarray
            Second scenario (features x samples)
        weights1 : np.ndarray, optional
            Weights for first scenario
        weights2 : np.ndarray, optional
            Weights for second scenario

        Returns
        -------
        float
            Distance between scenarios
        """
        return self.wasserstein.compute_nd(scenario1, scenario2, weights1, weights2)

    def detect_regime_shift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Detect if a regime shift has occurred.

        Parameters
        ----------
        baseline : np.ndarray
            Baseline scenario distribution
        current : np.ndarray
            Current scenario distribution
        threshold : float
            Threshold for detecting shift

        Returns
        -------
        tuple
            (shift_detected, distance)
        """
        distance = self.compare_scenarios(baseline, current)
        shift_detected = distance > threshold

        return shift_detected, distance

    def quantify_shock_impact(
        self,
        pre_shock: np.ndarray,
        post_shock: np.ndarray
    ) -> dict:
        """
        Quantify the impact of a shock event.

        Parameters
        ----------
        pre_shock : np.ndarray
            Pre-shock scenario distribution
        post_shock : np.ndarray
            Post-shock scenario distribution

        Returns
        -------
        dict
            Dictionary with impact metrics
        """
        distance = self.compare_scenarios(pre_shock, post_shock)

        # Compute additional metrics
        mean_shift = np.linalg.norm(np.mean(post_shock, axis=0) - np.mean(pre_shock, axis=0))
        variance_change = np.abs(np.var(post_shock) - np.var(pre_shock))

        return {
            'wasserstein_distance': distance,
            'mean_shift': mean_shift,
            'variance_change': variance_change,
            'impact_magnitude': distance * mean_shift
        }

    def compute_scenario_trajectory(
        self,
        scenarios: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute trajectory of scenarios over time.

        Parameters
        ----------
        scenarios : list of np.ndarray
            Time series of scenarios

        Returns
        -------
        np.ndarray
            Array of distances between consecutive scenarios
        """
        n_scenarios = len(scenarios)
        distances = np.zeros(n_scenarios - 1)

        for i in range(n_scenarios - 1):
            distances[i] = self.compare_scenarios(scenarios[i], scenarios[i + 1])

        return distances

    def logistics_optimal_transport(
        self,
        supply: np.ndarray,
        demand: np.ndarray,
        supply_locations: np.ndarray,
        demand_locations: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Solve logistics problem using optimal transport.

        Parameters
        ----------
        supply : np.ndarray
            Supply amounts at each location
        demand : np.ndarray
            Demand amounts at each location
        supply_locations : np.ndarray
            Coordinates of supply locations
        demand_locations : np.ndarray
            Coordinates of demand locations

        Returns
        -------
        tuple
            (transport_plan, total_cost)
        """
        if not HAS_POT:
            raise ImportError("POT library required for logistics optimization")

        # Normalize supply and demand
        supply_norm = supply / supply.sum()
        demand_norm = demand / demand.sum()

        # Compute cost matrix (distances)
        M = cdist(supply_locations, demand_locations, metric=self.wasserstein.metric)

        # Compute optimal transport plan
        transport_plan = ot.emd(supply_norm, demand_norm, M)
        total_cost = np.sum(transport_plan * M)

        # Scale back to original quantities
        transport_plan *= supply.sum()

        return transport_plan, total_cost
