"""
Do-Calculus Module - Intervention Reasoning

Implements Pearl's do-calculus for counterfactual analysis and policy simulation.

Instead of just forecasting "what will happen," this module enables:
- "What if the U.S. sanctions X?"
- "What if China mobilizes?"
- "What if NATO deploys troops?"
- "What if an election is rigged?"

This is the foundation for counterfactual geopolitics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple, Any
from ..models.causal_graph import CausalGraph, StructuralCausalModel


class DoCalculus:
    """
    Implement Pearl's do-calculus for causal inference.

    The do-calculus provides rules for transforming interventional
    distributions into observational ones, enabling causal effect
    estimation from observational data.
    """

    def __init__(self, causal_graph: CausalGraph):
        """
        Initialize do-calculus engine.

        Parameters
        ----------
        causal_graph : CausalGraph
            Causal graph structure
        """
        self.graph = causal_graph

    def is_identifiable(
        self,
        treatment: str,
        outcome: str,
        confounders: Optional[Set[str]] = None
    ) -> bool:
        """
        Check if causal effect is identifiable.

        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable
        confounders : Set[str], optional
            Known confounders

        Returns
        -------
        bool
            True if effect is identifiable
        """
        # Basic check: are treatment and outcome d-separated after intervention?
        # This is a simplified version

        # Get all backdoor paths
        backdoor_paths = self._get_backdoor_paths(treatment, outcome)

        if len(backdoor_paths) == 0:
            # No backdoor paths, effect is identifiable
            return True

        if confounders is not None:
            # Check if confounders block all backdoor paths
            return self._blocks_backdoor_paths(backdoor_paths, confounders)

        return False

    def _get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        Get all backdoor paths from treatment to outcome.

        A backdoor path is a path from treatment to outcome that
        starts with an arrow into the treatment.

        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable

        Returns
        -------
        List[List[str]]
            List of backdoor paths
        """
        import networkx as nx

        backdoor_paths = []

        # Get all simple paths from treatment to outcome
        try:
            all_paths = list(nx.all_simple_paths(
                self.graph.graph.to_undirected(),
                treatment,
                outcome
            ))
        except nx.NetworkXNoPath:
            return []

        # Filter for backdoor paths
        for path in all_paths:
            if len(path) > 2:  # Must have intermediate nodes
                # Check if first edge goes into treatment
                second_node = path[1]
                if self.graph.graph.has_edge(second_node, treatment):
                    backdoor_paths.append(path)

        return backdoor_paths

    def _blocks_backdoor_paths(
        self,
        paths: List[List[str]],
        conditioning_set: Set[str]
    ) -> bool:
        """
        Check if conditioning set blocks all backdoor paths.

        Parameters
        ----------
        paths : List[List[str]]
            Backdoor paths
        conditioning_set : Set[str]
            Variables to condition on

        Returns
        -------
        bool
            True if all paths are blocked
        """
        for path in paths:
            if not self._is_path_blocked(path, conditioning_set):
                return False
        return True

    def _is_path_blocked(self, path: List[str], conditioning_set: Set[str]) -> bool:
        """
        Check if a path is blocked by conditioning set.

        Parameters
        ----------
        path : List[str]
            Path to check
        conditioning_set : Set[str]
            Conditioning set

        Returns
        -------
        bool
            True if path is blocked
        """
        # Simplified version: check if any non-collider in path is in conditioning set
        for node in path[1:-1]:  # Exclude endpoints
            if node in conditioning_set:
                # Check if it's a collider
                idx = path.index(node)
                prev_node = path[idx - 1]
                next_node = path[idx + 1]

                # It's a collider if both edges point to it
                is_collider = (
                    self.graph.graph.has_edge(prev_node, node) and
                    self.graph.graph.has_edge(next_node, node)
                )

                if not is_collider:
                    return True

        return False

    def find_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        method: str = 'backdoor'
    ) -> Set[str]:
        """
        Find valid adjustment set for identifying causal effect.

        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable
        method : str
            Method to use ('backdoor', 'minimal')

        Returns
        -------
        Set[str]
            Valid adjustment set
        """
        if method == 'backdoor':
            return self._backdoor_adjustment_set(treatment, outcome)
        elif method == 'minimal':
            return self._minimal_adjustment_set(treatment, outcome)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _backdoor_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        """
        Find backdoor adjustment set.

        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable

        Returns
        -------
        Set[str]
            Backdoor adjustment set
        """
        # Get all parents of treatment (excluding outcome's descendants)
        parents = set(self.graph.get_parents(treatment))

        # Remove outcome and its descendants
        outcome_descendants = self.graph.get_descendants(outcome)
        adjustment_set = parents - outcome_descendants - {outcome}

        return adjustment_set

    def _minimal_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        """
        Find minimal adjustment set.

        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable

        Returns
        -------
        Set[str]
            Minimal adjustment set
        """
        # Start with backdoor set
        backdoor_set = self._backdoor_adjustment_set(treatment, outcome)

        # Try removing variables one by one
        minimal_set = backdoor_set.copy()

        for var in backdoor_set:
            candidate_set = minimal_set - {var}
            backdoor_paths = self._get_backdoor_paths(treatment, outcome)

            if self._blocks_backdoor_paths(backdoor_paths, candidate_set):
                minimal_set = candidate_set

        return minimal_set

    def compute_ate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: Optional[Set[str]] = None
    ) -> float:
        """
        Compute Average Treatment Effect (ATE).

        ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

        Parameters
        ----------
        data : pd.DataFrame
            Observational data
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable
        adjustment_set : Set[str], optional
            Variables to adjust for

        Returns
        -------
        float
            Average Treatment Effect
        """
        if adjustment_set is None:
            adjustment_set = self.find_adjustment_set(treatment, outcome)

        # Stratification estimator
        if len(adjustment_set) == 0:
            # No confounding
            treated = data[data[treatment] == 1][outcome].mean()
            control = data[data[treatment] == 0][outcome].mean()
            return treated - control

        # With adjustment
        # Group by adjustment variables
        adjustment_vars = list(adjustment_set)

        ate = 0.0
        for strata, group in data.groupby(adjustment_vars):
            if len(group) > 0:
                # Compute effect in this stratum
                treated = group[group[treatment] == 1][outcome].mean()
                control = group[group[treatment] == 0][outcome].mean()

                if not np.isnan(treated) and not np.isnan(control):
                    strata_effect = treated - control
                    strata_weight = len(group) / len(data)
                    ate += strata_effect * strata_weight

        return ate


class InterventionSimulator:
    """
    Simulate policy interventions using structural causal models.

    This class provides high-level interface for testing
    "what if" scenarios in geopolitical contexts.
    """

    def __init__(self, scm: StructuralCausalModel):
        """
        Initialize intervention simulator.

        Parameters
        ----------
        scm : StructuralCausalModel
            Structural causal model
        """
        self.scm = scm
        self.do_calculus = DoCalculus(scm.graph)

    def simulate_intervention(
        self,
        intervention: Dict[str, float],
        n_samples: int = 1000,
        outcomes: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate an intervention.

        Parameters
        ----------
        intervention : dict
            Intervention specification {variable: value}
        n_samples : int
            Number of Monte Carlo samples
        outcomes : List[str], optional
            Outcome variables to track

        Returns
        -------
        dict
            Simulated outcomes
        """
        # Sample from intervened distribution
        samples = self.scm.sample(n_samples=n_samples, interventions=intervention)

        if outcomes is not None:
            samples = {k: v for k, v in samples.items() if k in outcomes}

        return samples

    def compare_interventions(
        self,
        interventions: List[Dict[str, float]],
        outcome: str,
        n_samples: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple interventions.

        Parameters
        ----------
        interventions : List[dict]
            List of interventions to compare
        outcome : str
            Outcome variable to compare
        n_samples : int
            Number of samples per intervention

        Returns
        -------
        dict
            Comparison results
        """
        results = {}

        for i, intervention in enumerate(interventions):
            samples = self.simulate_intervention(intervention, n_samples, [outcome])
            outcome_samples = samples[outcome]

            results[f"intervention_{i}"] = {
                'intervention': intervention,
                'mean': np.mean(outcome_samples),
                'std': np.std(outcome_samples),
                'median': np.median(outcome_samples),
                'q25': np.percentile(outcome_samples, 25),
                'q75': np.percentile(outcome_samples, 75)
            }

        return results

    def optimal_intervention(
        self,
        target_var: str,
        intervention_vars: List[str],
        intervention_ranges: Dict[str, Tuple[float, float]],
        objective: str = 'maximize',
        n_trials: int = 100,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Find optimal intervention to achieve target.

        Parameters
        ----------
        target_var : str
            Target variable to optimize
        intervention_vars : List[str]
            Variables that can be intervened on
        intervention_ranges : dict
            Ranges for each intervention variable
        objective : str
            'maximize' or 'minimize'
        n_trials : int
            Number of random trials
        n_samples : int
            Samples per trial

        Returns
        -------
        dict
            Optimal intervention and results
        """
        best_intervention = None
        best_value = float('-inf') if objective == 'maximize' else float('inf')

        for _ in range(n_trials):
            # Sample random intervention
            intervention = {}
            for var in intervention_vars:
                low, high = intervention_ranges[var]
                intervention[var] = np.random.uniform(low, high)

            # Simulate
            samples = self.simulate_intervention(intervention, n_samples, [target_var])
            mean_value = np.mean(samples[target_var])

            # Update best
            if objective == 'maximize':
                if mean_value > best_value:
                    best_value = mean_value
                    best_intervention = intervention
            else:
                if mean_value < best_value:
                    best_value = mean_value
                    best_intervention = intervention

        return {
            'optimal_intervention': best_intervention,
            'optimal_value': best_value,
            'objective': objective
        }

    def counterfactual_analysis(
        self,
        observed: Dict[str, float],
        intervention: Dict[str, float],
        outcome: str
    ) -> Dict[str, float]:
        """
        Perform counterfactual analysis.

        "Given that we observed X, what would have happened if we had done Y?"

        Parameters
        ----------
        observed : dict
            Observed values
        intervention : dict
            Counterfactual intervention
        outcome : str
            Outcome variable

        Returns
        -------
        dict
            Counterfactual results
        """
        counterfactual = self.scm.compute_counterfactual(observed, intervention)

        return {
            'observed_outcome': observed.get(outcome, None),
            'counterfactual_outcome': counterfactual.get(outcome, None),
            'effect': counterfactual.get(outcome, 0) - observed.get(outcome, 0)
        }
