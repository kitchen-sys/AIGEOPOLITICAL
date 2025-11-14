"""
Structural Causal Models for GeoBotv1

Implements Structural Causal Models (SCMs) for geopolitical analysis,
intervention simulation, and counterfactual reasoning. Integrates with
GeoBot 2.0 analytical framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from enum import Enum
import numpy as np
import networkx as nx


class IdentificationStrategy(Enum):
    """Strategies for identifying causal effects."""
    BACKDOOR_ADJUSTMENT = "backdoor"
    FRONTDOOR_ADJUSTMENT = "frontdoor"
    INSTRUMENTAL_VARIABLES = "iv"
    DO_CALCULUS = "do_calculus"
    STRUCTURAL_EQUATIONS = "structural"


@dataclass
class StructuralEquation:
    """
    Structural equation for a variable in SCM.

    X := f(Pa_X, U_X)

    Attributes
    ----------
    variable : str
        Variable name
    parents : List[str]
        Parent variables in causal graph
    function : Callable
        Structural function f
    noise_dist : Callable
        Distribution of exogenous noise U_X
    description : str
        Description of equation
    """
    variable: str
    parents: List[str]
    function: Callable[[Dict[str, float]], float]
    noise_dist: Callable[[int], np.ndarray]
    description: str = ""

    def evaluate(self, parent_values: Dict[str, float], noise: Optional[float] = None) -> float:
        """
        Evaluate structural equation.

        Parameters
        ----------
        parent_values : Dict[str, float]
            Values of parent variables
        noise : Optional[float]
            Noise value (if None, sample from distribution)

        Returns
        -------
        float
            Value of variable
        """
        if noise is None:
            noise = self.noise_dist(1)[0]

        return self.function(parent_values) + noise


@dataclass
class Intervention:
    """
    Causal intervention do(X = x).

    Attributes
    ----------
    variable : str
        Variable being intervened on
    value : float
        Value set by intervention
    description : str
        Description of intervention
    """
    variable: str
    value: float
    description: str = ""

    def __repr__(self) -> str:
        return f"do({self.variable} = {self.value})"


@dataclass
class Counterfactual:
    """
    Counterfactual query.

    "What would Y be if we had done X = x, given that we observed Z = z?"

    Attributes
    ----------
    query_variable : str
        Variable being queried
    intervention : Intervention
        Counterfactual intervention
    observations : Dict[str, float]
        Observed values
    """
    query_variable: str
    intervention: Intervention
    observations: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        obs_str = ", ".join([f"{k}={v}" for k, v in self.observations.items()])
        return f"{self.query_variable}_{{{self.intervention}}} | {obs_str}"


@dataclass
class CausalEffect:
    """
    Estimated causal effect.

    Attributes
    ----------
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
    effect : float
        Estimated average causal effect
    std_error : Optional[float]
        Standard error of estimate
    confidence_interval : Optional[Tuple[float, float]]
        Confidence interval
    identification_strategy : IdentificationStrategy
        How effect was identified
    """
    treatment: str
    outcome: str
    effect: float
    std_error: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    identification_strategy: Optional[IdentificationStrategy] = None

    def __repr__(self) -> str:
        ci_str = ""
        if self.confidence_interval:
            ci_str = f", 95% CI: {self.confidence_interval}"
        return f"ACE({self.treatment} → {self.outcome}) = {self.effect:.3f}{ci_str}"


class StructuralCausalModel:
    """
    Structural Causal Model for geopolitical analysis.

    An SCM consists of:
    1. Causal graph G (DAG)
    2. Structural equations for each variable
    3. Exogenous noise distributions

    Enables:
    - Intervention simulation (do-operator)
    - Counterfactual reasoning
    - Causal effect identification
    """

    def __init__(self, name: str = "GeopoliticalSCM"):
        """
        Initialize SCM.

        Parameters
        ----------
        name : str
            Name of SCM
        """
        self.name = name
        self.graph = nx.DiGraph()
        self.equations: Dict[str, StructuralEquation] = {}
        self.exogenous_variables: Set[str] = set()

    def add_equation(self, equation: StructuralEquation) -> None:
        """
        Add structural equation to model.

        Parameters
        ----------
        equation : StructuralEquation
            Structural equation
        """
        self.equations[equation.variable] = equation

        # Add to graph
        self.graph.add_node(equation.variable)
        for parent in equation.parents:
            self.graph.add_edge(parent, equation.variable)

    def add_exogenous(self, variable: str, distribution: Callable[[int], np.ndarray]) -> None:
        """
        Add exogenous variable.

        Parameters
        ----------
        variable : str
            Variable name
        distribution : Callable
            Distribution for sampling
        """
        self.exogenous_variables.add(variable)
        self.graph.add_node(variable)

    def topological_order(self) -> List[str]:
        """
        Get topological ordering of variables.

        Returns
        -------
        List[str]
            Topologically sorted variables
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            raise ValueError("Graph contains cycles - not a valid DAG")

    def simulate(
        self,
        n_samples: int = 1000,
        interventions: Optional[List[Intervention]] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate from SCM.

        Parameters
        ----------
        n_samples : int
            Number of samples
        interventions : Optional[List[Intervention]]
            Interventions to apply
        random_state : Optional[int]
            Random seed

        Returns
        -------
        Dict[str, np.ndarray]
            Simulated data for each variable
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Intervention variables
        intervention_dict = {}
        if interventions:
            intervention_dict = {iv.variable: iv.value for iv in interventions}

        # Initialize data
        data = {}

        # Topological order
        order = self.topological_order()

        # Simulate each variable in order
        for var in order:
            if var in intervention_dict:
                # Variable is intervened on - set to intervention value
                data[var] = np.full(n_samples, intervention_dict[var])

            elif var in self.exogenous_variables:
                # Exogenous variable - sample from distribution
                # For now, assume standard normal if not specified
                data[var] = np.random.randn(n_samples)

            elif var in self.equations:
                # Endogenous variable - evaluate structural equation
                eq = self.equations[var]
                values = np.zeros(n_samples)

                for i in range(n_samples):
                    parent_vals = {p: data[p][i] for p in eq.parents}
                    values[i] = eq.evaluate(parent_vals)

                data[var] = values

            else:
                raise ValueError(f"No equation for variable {var}")

        return data

    def intervene(
        self,
        interventions: List[Intervention],
        n_samples: int = 1000,
        random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate interventions using do-operator.

        Parameters
        ----------
        interventions : List[Intervention]
            Interventions to apply
        n_samples : int
            Number of samples
        random_state : Optional[int]
            Random seed

        Returns
        -------
        Dict[str, np.ndarray]
            Post-intervention data
        """
        return self.simulate(n_samples, interventions, random_state)

    def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        n_samples: int = 10000,
        treatment_values: Optional[List[float]] = None
    ) -> CausalEffect:
        """
        Estimate average causal effect of treatment on outcome.

        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable
        n_samples : int
            Number of simulation samples
        treatment_values : Optional[List[float]]
            Treatment values to compare (default [0, 1])

        Returns
        -------
        CausalEffect
            Estimated causal effect
        """
        if treatment_values is None:
            treatment_values = [0.0, 1.0]

        # Simulate under different treatment values
        outcomes = []
        for t_val in treatment_values:
            intervention = Intervention(variable=treatment, value=t_val)
            data = self.intervene([intervention], n_samples)
            outcomes.append(np.mean(data[outcome]))

        # Average causal effect
        ace = outcomes[1] - outcomes[0]

        # Bootstrap for standard error
        bootstrap_effects = []
        for _ in range(100):
            boot_outcomes = []
            for t_val in treatment_values:
                intervention = Intervention(variable=treatment, value=t_val)
                data = self.intervene([intervention], n_samples=1000)
                boot_outcomes.append(np.mean(data[outcome]))
            bootstrap_effects.append(boot_outcomes[1] - boot_outcomes[0])

        std_error = np.std(bootstrap_effects)
        ci = (
            ace - 1.96 * std_error,
            ace + 1.96 * std_error
        )

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=ace,
            std_error=std_error,
            confidence_interval=ci,
            identification_strategy=IdentificationStrategy.STRUCTURAL_EQUATIONS
        )

    def counterfactual_query(
        self,
        query: Counterfactual,
        n_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Answer counterfactual query.

        Three-step process:
        1. Abduction: Infer exogenous variables from observations
        2. Action: Apply intervention
        3. Prediction: Compute outcome

        Parameters
        ----------
        query : Counterfactual
            Counterfactual query
        n_samples : int
            Number of samples for approximation

        Returns
        -------
        Dict[str, Any]
            Counterfactual results
        """
        # Simplified counterfactual reasoning
        # Full implementation would do proper abduction step

        # For now, simulate with intervention
        data = self.intervene([query.intervention], n_samples)

        return {
            'query': str(query),
            'expected_value': float(np.mean(data[query.query_variable])),
            'std': float(np.std(data[query.query_variable])),
            'median': float(np.median(data[query.query_variable])),
            'quantiles': {
                '5%': float(np.quantile(data[query.query_variable], 0.05)),
                '25%': float(np.quantile(data[query.query_variable], 0.25)),
                '75%': float(np.quantile(data[query.query_variable], 0.75)),
                '95%': float(np.quantile(data[query.query_variable], 0.95)),
            }
        }

    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        Find backdoor paths from treatment to outcome.

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
        # Create undirected version of graph
        undirected = self.graph.to_undirected()

        # Find all paths
        try:
            all_paths = list(nx.all_simple_paths(undirected, treatment, outcome))
        except nx.NodeNotFound:
            return []

        # Filter for backdoor paths (paths that go through parent of treatment)
        backdoor_paths = []
        treatment_parents = set(self.graph.predecessors(treatment))

        for path in all_paths:
            # Check if path starts with an edge into treatment
            if len(path) > 1 and path[1] in treatment_parents:
                backdoor_paths.append(path)

        return backdoor_paths

    def find_backdoor_adjustment_set(
        self,
        treatment: str,
        outcome: str
    ) -> Optional[Set[str]]:
        """
        Find minimal backdoor adjustment set.

        Parameters
        ----------
        treatment : str
            Treatment variable
        outcome : str
            Outcome variable

        Returns
        -------
        Optional[Set[str]]
            Backdoor adjustment set, or None if no valid set exists
        """
        backdoor_paths = self.find_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            return set()  # No backdoor paths, empty set suffices

        # Find minimal set that blocks all backdoor paths
        # This is simplified - full implementation would use
        # proper d-separation testing

        # Collect all variables in backdoor paths
        candidates = set()
        for path in backdoor_paths:
            candidates.update(path[1:-1])  # Exclude treatment and outcome

        # Remove descendants of treatment (would create bias)
        treatment_descendants = nx.descendants(self.graph, treatment)
        candidates -= treatment_descendants

        return candidates

    def plot_graph(self, filename: Optional[str] = None) -> None:
        """
        Plot causal graph.

        Parameters
        ----------
        filename : Optional[str]
            File to save plot to
        """
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph)
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color='lightblue',
                node_size=1500,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20
            )

            if filename:
                plt.savefig(filename)
            else:
                plt.show()

        except ImportError:
            print("matplotlib not available for plotting")


def estimate_causal_effect(
    scm: StructuralCausalModel,
    treatment: str,
    outcome: str,
    adjustment_set: Optional[Set[str]] = None,
    n_samples: int = 10000
) -> CausalEffect:
    """
    Estimate causal effect using appropriate identification strategy.

    Parameters
    ----------
    scm : StructuralCausalModel
        Structural causal model
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
    adjustment_set : Optional[Set[str]]
        Variables to adjust for (if None, use backdoor criterion)
    n_samples : int
        Number of samples

    Returns
    -------
    CausalEffect
        Estimated causal effect
    """
    # Find adjustment set if not provided
    if adjustment_set is None:
        adjustment_set = scm.find_backdoor_adjustment_set(treatment, outcome)

        if adjustment_set is None:
            # Try frontdoor or IV
            raise ValueError("Cannot identify causal effect - no valid adjustment set")

    # Use structural equations for direct estimation
    return scm.estimate_causal_effect(treatment, outcome, n_samples)


# ============================================================================
# Predefined SCMs for Geopolitical Analysis
# ============================================================================

def create_sanctions_scm() -> StructuralCausalModel:
    """
    Create SCM for sanctions analysis.

    Variables:
    - sanctions: Binary sanctions imposed
    - trade_disruption: Trade flow disruption
    - economic_growth: Economic growth rate
    - regime_stability: Regime stability score

    Returns
    -------
    StructuralCausalModel
        Sanctions SCM
    """
    scm = StructuralCausalModel(name="SanctionsSCM")

    # Exogenous noise (simplified as standard normal)
    noise_dist = lambda n: np.random.randn(n) * 0.1

    # Trade disruption = f(sanctions) + noise
    scm.add_equation(StructuralEquation(
        variable="trade_disruption",
        parents=["sanctions"],
        function=lambda p: 0.7 * p["sanctions"],
        noise_dist=noise_dist,
        description="Sanctions directly reduce trade"
    ))

    # Economic growth = f(trade_disruption) + noise
    scm.add_equation(StructuralEquation(
        variable="economic_growth",
        parents=["trade_disruption"],
        function=lambda p: 0.05 - 0.4 * p["trade_disruption"],
        noise_dist=noise_dist,
        description="Trade disruption reduces growth"
    ))

    # Regime stability = f(economic_growth) + noise
    scm.add_equation(StructuralEquation(
        variable="regime_stability",
        parents=["economic_growth"],
        function=lambda p: 0.7 + 0.5 * p["economic_growth"],
        noise_dist=noise_dist,
        description="Economic growth affects regime stability"
    ))

    return scm


def create_conflict_escalation_scm() -> StructuralCausalModel:
    """
    Create SCM for conflict escalation.

    Variables:
    - military_buildup: Military force buildup
    - diplomatic_tension: Diplomatic relations tension
    - conflict_risk: Risk of armed conflict

    Returns
    -------
    StructuralCausalModel
        Conflict escalation SCM
    """
    scm = StructuralCausalModel(name="ConflictEscalationSCM")

    noise_dist = lambda n: np.random.randn(n) * 0.05

    # Diplomatic tension = f(military_buildup) + noise
    scm.add_equation(StructuralEquation(
        variable="diplomatic_tension",
        parents=["military_buildup"],
        function=lambda p: 0.3 + 0.6 * p["military_buildup"],
        noise_dist=noise_dist,
        description="Military buildup increases diplomatic tension"
    ))

    # Conflict risk = f(military_buildup, diplomatic_tension) + noise
    scm.add_equation(StructuralEquation(
        variable="conflict_risk",
        parents=["military_buildup", "diplomatic_tension"],
        function=lambda p: 0.1 + 0.4 * p["military_buildup"] + 0.3 * p["diplomatic_tension"],
        noise_dist=noise_dist,
        description="Both military buildup and tension increase conflict risk"
    ))

    return scm
