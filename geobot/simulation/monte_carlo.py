"""
Monte Carlo Simulation Engine

Stochastic simulation for geopolitical forecasting with support for:
- Monte Carlo over causal graphs
- Agent-based Monte Carlo
- Stochastic war-gaming simulations
- Shock Monte Carlo (black swan simulation)

The more structural and stochastic your simulations, the more your engine
resembles a national-security world model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from ..models.causal_graph import StructuralCausalModel
from ..core.scenario import Scenario, ScenarioDistribution


@dataclass
class SimulationConfig:
    """
    Configuration for Monte Carlo simulation.

    Attributes
    ----------
    n_simulations : int
        Number of Monte Carlo runs
    time_horizon : int
        Simulation time horizon
    random_seed : int
        Random seed for reproducibility
    parallel : bool
        Run simulations in parallel
    """
    n_simulations: int = 1000
    time_horizon: int = 100
    random_seed: Optional[int] = None
    parallel: bool = False


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for geopolitical forecasting.

    Supports various types of stochastic simulation including:
    - Basic Monte Carlo
    - Causal graph-based simulation
    - Shock simulation
    - Path-dependent simulation
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize Monte Carlo engine.

        Parameters
        ----------
        config : SimulationConfig, optional
            Simulation configuration
        """
        self.config = config or SimulationConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def run_basic_simulation(
        self,
        initial_state: Dict[str, float],
        transition_fn: Callable,
        noise_fn: Optional[Callable] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run basic Monte Carlo simulation.

        Parameters
        ----------
        initial_state : dict
            Initial state of the system
        transition_fn : callable
            State transition function
            Signature: f(state, t, noise) -> new_state
        noise_fn : callable, optional
            Noise generation function
            Signature: f(t) -> noise_dict

        Returns
        -------
        list
            List of simulation trajectories
        """
        trajectories = []

        for sim in range(self.config.n_simulations):
            trajectory = {var: np.zeros(self.config.time_horizon) for var in initial_state}

            # Initialize
            state = initial_state.copy()
            for var, val in state.items():
                trajectory[var][0] = val

            # Simulate forward
            for t in range(1, self.config.time_horizon):
                # Generate noise
                noise = noise_fn(t) if noise_fn else {}

                # Transition
                state = transition_fn(state, t, noise)

                # Record
                for var, val in state.items():
                    trajectory[var][t] = val

            trajectories.append(trajectory)

        return trajectories

    def run_causal_simulation(
        self,
        scm: StructuralCausalModel,
        initial_conditions: Optional[Dict[str, float]] = None,
        interventions: Optional[Dict[str, Dict[str, float]]] = None
    ) -> ScenarioDistribution:
        """
        Run Monte Carlo simulation over causal graph.

        Parameters
        ----------
        scm : StructuralCausalModel
            Structural causal model
        initial_conditions : dict, optional
            Initial conditions for some variables
        interventions : dict, optional
            Time-dependent interventions {time: {var: value}}

        Returns
        -------
        ScenarioDistribution
            Distribution of simulated scenarios
        """
        scenarios = []

        for sim in range(self.config.n_simulations):
            scenario_features = {}

            for t in range(self.config.time_horizon):
                # Get interventions at this time
                interv = interventions.get(t, {}) if interventions else {}

                # Sample from SCM
                samples = scm.sample(n_samples=1, interventions=interv)

                # Store
                for var, val in samples.items():
                    if var not in scenario_features:
                        scenario_features[var] = []
                    scenario_features[var].append(val[0])

            # Convert to arrays
            scenario_features = {k: np.array(v) for k, v in scenario_features.items()}

            # Create scenario
            scenario = Scenario(
                name=f"sim_{sim}",
                features=scenario_features,
                probability=1.0 / self.config.n_simulations
            )
            scenarios.append(scenario)

        return ScenarioDistribution(scenarios)

    def run_path_dependent_simulation(
        self,
        initial_state: Dict[str, float],
        transition_fn: Callable,
        decision_points: List[int],
        decision_fn: Callable
    ) -> Dict[str, Any]:
        """
        Run path-dependent simulation with decision points.

        This is useful for war-gaming and strategic scenarios where
        decisions depend on the current state.

        Parameters
        ----------
        initial_state : dict
            Initial state
        transition_fn : callable
            State transition function
        decision_points : list
            Time steps where decisions are made
        decision_fn : callable
            Decision function
            Signature: f(state, t) -> decision_dict

        Returns
        -------
        dict
            Simulation results with decision branches
        """
        trajectories = []
        decisions = []

        for sim in range(self.config.n_simulations):
            trajectory = {var: np.zeros(self.config.time_horizon) for var in initial_state}
            sim_decisions = []

            state = initial_state.copy()
            for var, val in state.items():
                trajectory[var][0] = val

            for t in range(1, self.config.time_horizon):
                # Check for decision point
                if t in decision_points:
                    decision = decision_fn(state, t)
                    sim_decisions.append((t, decision))
                    # Apply decision effects
                    for var, change in decision.items():
                        state[var] = state.get(var, 0) + change

                # Transition
                noise = {var: np.random.normal(0, 0.1) for var in state}
                state = transition_fn(state, t, noise)

                # Record
                for var, val in state.items():
                    trajectory[var][t] = val

            trajectories.append(trajectory)
            decisions.append(sim_decisions)

        return {
            'trajectories': trajectories,
            'decisions': decisions
        }

    def compute_statistics(
        self,
        trajectories: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute statistics across Monte Carlo trajectories.

        Parameters
        ----------
        trajectories : list
            List of simulation trajectories

        Returns
        -------
        dict
            Statistics for each variable
        """
        if len(trajectories) == 0:
            return {}

        variables = list(trajectories[0].keys())
        stats = {}

        for var in variables:
            # Stack trajectories
            data = np.array([traj[var] for traj in trajectories])

            stats[var] = {
                'mean': np.mean(data, axis=0),
                'median': np.median(data, axis=0),
                'std': np.std(data, axis=0),
                'q5': np.percentile(data, 5, axis=0),
                'q25': np.percentile(data, 25, axis=0),
                'q75': np.percentile(data, 75, axis=0),
                'q95': np.percentile(data, 95, axis=0),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0)
            }

        return stats

    def analyze_convergence(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        variable: str
    ) -> Dict[str, Any]:
        """
        Analyze convergence of Monte Carlo simulation.

        Parameters
        ----------
        trajectories : list
            Simulation trajectories
        variable : str
            Variable to analyze

        Returns
        -------
        dict
            Convergence metrics
        """
        data = np.array([traj[variable] for traj in trajectories])

        # Compute running mean
        n_sims = len(trajectories)
        running_means = []

        for n in range(10, n_sims, 10):
            running_means.append(np.mean(data[:n], axis=0))

        # Compute standard error
        se = np.std(data, axis=0) / np.sqrt(n_sims)

        return {
            'running_means': running_means,
            'standard_error': se,
            'converged': np.all(se < 0.01)  # Arbitrary threshold
        }


class ShockSimulator:
    """
    Simulate black swan events and shocks in geopolitical scenarios.

    This class specializes in modeling rare, high-impact events
    that are critical for risk assessment.
    """

    def __init__(self, mc_engine: Optional[MonteCarloEngine] = None):
        """
        Initialize shock simulator.

        Parameters
        ----------
        mc_engine : MonteCarloEngine, optional
            Monte Carlo engine to use
        """
        self.mc_engine = mc_engine or MonteCarloEngine()

    def generate_shock_scenarios(
        self,
        baseline_scenario: Dict[str, float],
        shock_types: List[Dict[str, Any]],
        shock_probabilities: List[float]
    ) -> ScenarioDistribution:
        """
        Generate scenarios including shock events.

        Parameters
        ----------
        baseline_scenario : dict
            Baseline scenario without shocks
        shock_types : list
            List of shock specifications
        shock_probabilities : list
            Probability of each shock type

        Returns
        -------
        ScenarioDistribution
            Distribution including shock scenarios
        """
        scenarios = []

        for sim in range(self.mc_engine.config.n_simulations):
            # Check if shock occurs
            shock_occurred = np.random.random() < sum(shock_probabilities)

            if shock_occurred:
                # Sample shock type
                shock_idx = np.random.choice(len(shock_types), p=np.array(shock_probabilities) / sum(shock_probabilities))
                shock = shock_types[shock_idx]

                # Apply shock
                scenario_state = baseline_scenario.copy()
                for var, impact in shock['impacts'].items():
                    scenario_state[var] = scenario_state.get(var, 0) + impact

                prob = shock_probabilities[shock_idx]
            else:
                scenario_state = baseline_scenario.copy()
                prob = 1.0 - sum(shock_probabilities)

            # Create scenario
            scenario = Scenario(
                name=f"shock_sim_{sim}",
                features={k: np.array([v]) for k, v in scenario_state.items()},
                probability=prob / self.mc_engine.config.n_simulations
            )
            scenarios.append(scenario)

        return ScenarioDistribution(scenarios)

    def simulate_cascading_failure(
        self,
        initial_failure: str,
        dependency_graph: Dict[str, List[str]],
        failure_probabilities: Dict[str, float],
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Simulate cascading failures in interconnected systems.

        Parameters
        ----------
        initial_failure : str
            Initial component that fails
        dependency_graph : dict
            Dependency relationships {component: [dependent_components]}
        failure_probabilities : dict
            Conditional failure probabilities
        n_simulations : int
            Number of simulations

        Returns
        -------
        dict
            Cascading failure analysis
        """
        cascade_results = []

        for _ in range(n_simulations):
            failed = {initial_failure}
            newly_failed = {initial_failure}

            # Simulate cascade
            max_iterations = 100
            for iteration in range(max_iterations):
                if len(newly_failed) == 0:
                    break

                current_newly_failed = set()

                for failed_component in newly_failed:
                    # Get dependent components
                    dependents = dependency_graph.get(failed_component, [])

                    for dependent in dependents:
                        if dependent not in failed:
                            # Check if it fails
                            p_fail = failure_probabilities.get(dependent, 0.5)
                            if np.random.random() < p_fail:
                                current_newly_failed.add(dependent)
                                failed.add(dependent)

                newly_failed = current_newly_failed

            cascade_results.append({
                'total_failures': len(failed),
                'failed_components': failed,
                'iterations': iteration + 1
            })

        return {
            'simulations': cascade_results,
            'mean_failures': np.mean([r['total_failures'] for r in cascade_results]),
            'max_failures': max([r['total_failures'] for r in cascade_results]),
            'failure_probability': {
                comp: np.mean([comp in r['failed_components'] for r in cascade_results])
                for comp in set().union(*[r['failed_components'] for r in cascade_results])
            }
        }

    def simulate_tail_risk(
        self,
        distribution: Callable,
        threshold: float,
        n_samples: int = 100000
    ) -> Dict[str, float]:
        """
        Simulate and analyze tail risk.

        Parameters
        ----------
        distribution : callable
            Distribution to sample from
        threshold : float
            Threshold for tail event
        n_samples : int
            Number of samples

        Returns
        -------
        dict
            Tail risk metrics
        """
        samples = distribution(n_samples)

        # Compute tail metrics
        exceed_threshold = samples > threshold
        tail_probability = np.mean(exceed_threshold)

        if tail_probability > 0:
            tail_samples = samples[exceed_threshold]
            conditional_mean = np.mean(tail_samples)
            conditional_std = np.std(tail_samples)
        else:
            conditional_mean = None
            conditional_std = None

        return {
            'tail_probability': tail_probability,
            'var_95': np.percentile(samples, 95),
            'var_99': np.percentile(samples, 99),
            'cvar_95': np.mean(samples[samples > np.percentile(samples, 95)]),
            'cvar_99': np.mean(samples[samples > np.percentile(samples, 99)]),
            'conditional_mean': conditional_mean,
            'conditional_std': conditional_std,
            'max_loss': np.max(samples)
        }

    def stress_test(
        self,
        baseline_state: Dict[str, float],
        transition_fn: Callable,
        stress_scenarios: List[Dict[str, float]],
        time_horizon: int = 50
    ) -> Dict[str, Any]:
        """
        Perform stress testing under extreme scenarios.

        Parameters
        ----------
        baseline_state : dict
            Baseline state
        transition_fn : callable
            State transition function
        stress_scenarios : list
            List of stress scenarios to test
        time_horizon : int
            Simulation horizon

        Returns
        -------
        dict
            Stress test results
        """
        results = {}

        for i, stress in enumerate(stress_scenarios):
            # Simulate under stress
            state = baseline_state.copy()

            # Apply stress shock at t=0
            for var, shock in stress.items():
                state[var] = state.get(var, 0) + shock

            # Simulate forward
            trajectory = {var: [val] for var, val in state.items()}

            for t in range(1, time_horizon):
                noise = {var: np.random.normal(0, 0.05) for var in state}
                state = transition_fn(state, t, noise)

                for var, val in state.items():
                    trajectory[var].append(val)

            results[f'stress_{i}'] = {
                'scenario': stress,
                'trajectory': trajectory,
                'final_state': {var: vals[-1] for var, vals in trajectory.items()}
            }

        return results
