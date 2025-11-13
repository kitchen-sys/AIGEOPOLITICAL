"""
Data validation utilities
"""

from typing import Any
from ..core.scenario import Scenario
from ..models.causal_graph import CausalGraph


def validate_scenario(scenario: Scenario) -> bool:
    """
    Validate scenario structure.

    Parameters
    ----------
    scenario : Scenario
        Scenario to validate

    Returns
    -------
    bool
        True if valid
    """
    if not scenario.name:
        return False

    if not scenario.features:
        return False

    if scenario.probability < 0 or scenario.probability > 1:
        return False

    return True


def validate_causal_graph(graph: CausalGraph) -> bool:
    """
    Validate causal graph structure.

    Parameters
    ----------
    graph : CausalGraph
        Graph to validate

    Returns
    -------
    bool
        True if valid (is DAG)
    """
    import networkx as nx
    return nx.is_directed_acyclic_graph(graph.graph)
