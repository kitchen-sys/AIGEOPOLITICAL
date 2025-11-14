"""
Hawkes Process Wrapper for GeoBotv1

Provides convenient access to Hawkes processes for conflict dynamics
and event contagion modeling. This module re-exports functionality from
geobot.timeseries.point_processes for easier discoverability.

Hawkes processes are self-exciting point processes that model how
events cluster in time and how they can trigger subsequent events,
making them ideal for modeling:
- Conflict escalation dynamics
- Contagion between countries/regions
- Cascading geopolitical events
"""

# Import from timeseries.point_processes
from ..timeseries.point_processes import (
    UnivariateHawkesProcess,
    MultivariateHawkesProcess,
    ConflictContagionModel,
    HawkesParameters,
    HawkesFitResult,
    estimate_branching_ratio,
    detect_explosive_regime
)

__all__ = [
    "UnivariateHawkesProcess",
    "MultivariateHawkesProcess",
    "ConflictContagionModel",
    "HawkesParameters",
    "HawkesFitResult",
    "estimate_branching_ratio",
    "detect_explosive_regime",
    "HawkesSimulator",  # Additional convenience class below
]


class HawkesSimulator:
    """
    High-level interface for Hawkes process simulation.

    Provides simplified API for common Hawkes process use cases
    in geopolitical modeling.
    """

    def __init__(self, n_dimensions: int = 1):
        """
        Initialize Hawkes simulator.

        Parameters
        ----------
        n_dimensions : int
            Number of dimensions (countries, regions, etc.)
        """
        self.n_dimensions = n_dimensions

        if n_dimensions == 1:
            self.process = UnivariateHawkesProcess()
        else:
            self.process = MultivariateHawkesProcess(n_dimensions=n_dimensions)

    def fit(self, events, T: float = None, **kwargs):
        """
        Fit Hawkes process to event data.

        Parameters
        ----------
        events : Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]
            Event times
        T : float
            Observation window
        **kwargs
            Additional arguments passed to fit method

        Returns
        -------
        HawkesFitResult
            Fit results
        """
        return self.process.fit(events, T=T, **kwargs)

    def simulate(
        self,
        T: float,
        params: HawkesParameters,
        random_state: int = None,
        **kwargs
    ):
        """
        Simulate Hawkes process.

        Parameters
        ----------
        T : float
            Simulation time horizon
        params : HawkesParameters
            Process parameters
        random_state : int
            Random seed
        **kwargs
            Additional arguments

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            Simulated event times
        """
        return self.process.simulate(T, params, random_state=random_state, **kwargs)

    def predict_intensity(self, events, t: float, params: HawkesParameters):
        """
        Predict intensity at time t given past events.

        Parameters
        ----------
        events : Union[np.ndarray, List[np.ndarray]]
            Past event times
        t : float
            Time at which to predict
        params : HawkesParameters
            Process parameters

        Returns
        -------
        Union[float, np.ndarray]
            Predicted intensity
        """
        return self.process.intensity(events, t, params)

    def assess_stability(self, params: HawkesParameters) -> dict:
        """
        Assess stability of Hawkes process.

        Parameters
        ----------
        params : HawkesParameters
            Process parameters

        Returns
        -------
        dict
            Stability assessment with branching ratio and regime
        """
        branching_ratio = estimate_branching_ratio(params)
        is_explosive = detect_explosive_regime(params)

        return {
            'branching_ratio': branching_ratio,
            'is_explosive': is_explosive,
            'is_stable': branching_ratio < 1.0,
            'regime': 'supercritical (explosive)' if is_explosive else 'subcritical (stable)',
            'interpretation': (
                'Process is stable - events will not cascade indefinitely'
                if branching_ratio < 1.0
                else 'Process is explosive - events can trigger cascading escalation'
            )
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_conflict_contagion_analysis(
    events_by_country: dict,
    T: float,
    country_names: list = None
) -> dict:
    """
    Quick analysis of conflict contagion between countries.

    Parameters
    ----------
    events_by_country : dict
        Dictionary mapping country names to event times
    T : float
        Observation window
    country_names : list
        List of country names (if None, use dict keys)

    Returns
    -------
    dict
        Contagion analysis results
    """
    if country_names is None:
        country_names = list(events_by_country.keys())

    model = ConflictContagionModel(countries=country_names)
    result = model.fit(events_by_country, T=T)

    return {
        'fit_result': result,
        'most_contagious': result['most_contagious_source'],
        'most_vulnerable': result['most_vulnerable_target'],
        'total_events': {
            country: len(events_by_country[country])
            for country in country_names
        },
        'contagion_matrix': result['alpha_matrix'],
        'baseline_rates': result['mu'],
        'branching_ratio': estimate_branching_ratio(result['parameters']),
    }


def simulate_conflict_scenario(
    n_countries: int,
    baseline_rates: list,
    contagion_strength: float,
    T: float,
    random_state: int = None
) -> dict:
    """
    Simulate conflict scenario with specified parameters.

    Parameters
    ----------
    n_countries : int
        Number of countries
    baseline_rates : list
        Baseline conflict rates for each country
    contagion_strength : float
        Strength of cross-country contagion
    T : float
        Simulation horizon
    random_state : int
        Random seed

    Returns
    -------
    dict
        Simulation results
    """
    import numpy as np

    # Create contagion matrix
    alpha_matrix = np.full((n_countries, n_countries), contagion_strength)
    np.fill_diagonal(alpha_matrix, contagion_strength * 1.5)  # Self-excitation stronger

    # Create parameters
    params = HawkesParameters(
        mu=np.array(baseline_rates),
        alpha=alpha_matrix,
        beta=np.ones((n_countries, n_countries)) * 2.0  # Decay rate
    )

    # Simulate
    simulator = HawkesSimulator(n_dimensions=n_countries)
    events = simulator.simulate(T, params, random_state=random_state)

    # Assess stability
    stability = simulator.assess_stability(params)

    return {
        'events': events,
        'parameters': params,
        'stability': stability,
        'event_counts': [len(e) for e in events],
        'total_events': sum(len(e) for e in events),
        'average_rate': sum(len(e) for e in events) / T / n_countries,
    }
