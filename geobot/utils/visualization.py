"""
Visualization utilities for GeoBotv1
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def plot_scenario_distribution(
    scenarios: list,
    feature: str,
    output_path: Optional[str] = None
) -> None:
    """
    Plot distribution of scenarios for a feature.

    Parameters
    ----------
    scenarios : list
        List of scenarios
    feature : str
        Feature to plot
    output_path : str, optional
        Path to save plot
    """
    values = [s.features.get(feature, [0])[0] for s in scenarios]
    probabilities = [s.probability for s in scenarios]

    plt.figure(figsize=(10, 6))
    plt.hist(values, weights=probabilities, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Probability')
    plt.title(f'Distribution of {feature}')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_causal_graph(graph, output_path: Optional[str] = None) -> None:
    """
    Plot causal graph.

    Parameters
    ----------
    graph : CausalGraph
        Graph to plot
    output_path : str, optional
        Path to save plot
    """
    graph.visualize(output_path)
