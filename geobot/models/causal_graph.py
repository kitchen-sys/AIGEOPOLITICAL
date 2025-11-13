"""
Causal Graph Module - DAG Representation

Provides infrastructure for representing and analyzing causal relationships
in geopolitical systems using Directed Acyclic Graphs (DAGs).

This module answers:
- What causes conflict?
- What causes collapse?
- What causes escalation?
- What causes mobilization?
- What causes instability?

Critical for: Real forecasting of interventions, not just correlation-based guessing.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import json


@dataclass
class CausalEdge:
    """
    Represents a causal edge in the graph.

    Attributes
    ----------
    source : str
        Source node (cause)
    target : str
        Target node (effect)
    strength : float
        Strength of causal relationship (-1 to 1)
    confidence : float
        Confidence in this relationship (0 to 1)
    mechanism : str
        Description of causal mechanism
    """
    source: str
    target: str
    strength: float = 1.0
    confidence: float = 1.0
    mechanism: str = ""


class CausalGraph:
    """
    Directed Acyclic Graph (DAG) for causal relationships.

    This class provides the foundation for causal inference in geopolitical
    forecasting, ensuring that we understand what actually causes events
    rather than just observing correlations.
    """

    def __init__(self, name: str = "geopolitical_dag"):
        """
        Initialize causal graph.

        Parameters
        ----------
        name : str
            Name of the causal graph
        """
        self.name = name
        self.graph = nx.DiGraph()
        self.edges: List[CausalEdge] = []
        self.node_metadata: Dict[str, Dict[str, Any]] = {}

    def add_node(
        self,
        node: str,
        node_type: str = "variable",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a node to the causal graph.

        Parameters
        ----------
        node : str
            Node identifier
        node_type : str
            Type of node ('variable', 'event', 'policy', 'state')
        metadata : dict, optional
            Additional metadata for the node
        """
        self.graph.add_node(node)
        self.node_metadata[node] = {
            'type': node_type,
            'metadata': metadata or {}
        }

    def add_edge(
        self,
        source: str,
        target: str,
        strength: float = 1.0,
        confidence: float = 1.0,
        mechanism: str = ""
    ) -> None:
        """
        Add a causal edge to the graph.

        Parameters
        ----------
        source : str
            Source node (cause)
        target : str
            Target node (effect)
        strength : float
            Strength of causal relationship
        confidence : float
            Confidence in this relationship
        mechanism : str
            Description of causal mechanism
        """
        # Check for cycles
        if not self._would_create_cycle(source, target):
            self.graph.add_edge(source, target)
            edge = CausalEdge(source, target, strength, confidence, mechanism)
            self.edges.append(edge)
        else:
            raise ValueError(f"Adding edge {source} -> {target} would create a cycle")

    def remove_edge(self, source: str, target: str) -> None:
        """
        Remove a causal edge.

        Parameters
        ----------
        source : str
            Source node
        target : str
            Target node
        """
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """
        Check if adding an edge would create a cycle.

        Parameters
        ----------
        source : str
            Source node
        target : str
            Target node

        Returns
        -------
        bool
            True if edge would create cycle
        """
        # Add nodes if they don't exist
        if source not in self.graph:
            self.graph.add_node(source)
        if target not in self.graph:
            self.graph.add_node(target)

        # Temporarily add edge and check for cycles
        self.graph.add_edge(source, target)
        has_cycle = not nx.is_directed_acyclic_graph(self.graph)
        self.graph.remove_edge(source, target)

        return has_cycle

    def get_parents(self, node: str) -> List[str]:
        """
        Get direct parents (causes) of a node.

        Parameters
        ----------
        node : str
            Node identifier

        Returns
        -------
        List[str]
            List of parent nodes
        """
        return list(self.graph.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        """
        Get direct children (effects) of a node.

        Parameters
        ----------
        node : str
            Node identifier

        Returns
        -------
        List[str]
            List of child nodes
        """
        return list(self.graph.successors(node))

    def get_ancestors(self, node: str) -> Set[str]:
        """
        Get all ancestors (causes) of a node.

        Parameters
        ----------
        node : str
            Node identifier

        Returns
        -------
        Set[str]
            Set of ancestor nodes
        """
        return nx.ancestors(self.graph, node)

    def get_descendants(self, node: str) -> Set[str]:
        """
        Get all descendants (effects) of a node.

        Parameters
        ----------
        node : str
            Node identifier

        Returns
        -------
        Set[str]
            Set of descendant nodes
        """
        return nx.descendants(self.graph, node)

    def get_topological_order(self) -> List[str]:
        """
        Get topological ordering of nodes.

        This is useful for computing values in causal order.

        Returns
        -------
        List[str]
            Nodes in topological order
        """
        return list(nx.topological_sort(self.graph))

    def is_ancestor(self, node1: str, node2: str) -> bool:
        """
        Check if node1 is an ancestor of node2.

        Parameters
        ----------
        node1 : str
            Potential ancestor
        node2 : str
            Potential descendant

        Returns
        -------
        bool
            True if node1 is ancestor of node2
        """
        return node1 in self.get_ancestors(node2)

    def is_descendant(self, node1: str, node2: str) -> bool:
        """
        Check if node1 is a descendant of node2.

        Parameters
        ----------
        node1 : str
            Potential descendant
        node2 : str
            Potential ancestor

        Returns
        -------
        bool
            True if node1 is descendant of node2
        """
        return node1 in self.get_descendants(node2)

    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        Get Markov blanket of a node.

        The Markov blanket includes: parents, children, and co-parents
        (other parents of children).

        Parameters
        ----------
        node : str
            Node identifier

        Returns
        -------
        Set[str]
            Markov blanket nodes
        """
        parents = set(self.get_parents(node))
        children = set(self.get_children(node))

        # Get co-parents (parents of children)
        co_parents = set()
        for child in children:
            co_parents.update(self.get_parents(child))

        co_parents.discard(node)

        return parents | children | co_parents

    def d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        Test if X and Y are d-separated given Z.

        This is fundamental for determining conditional independence.

        Parameters
        ----------
        X : Set[str]
            First set of nodes
        Y : Set[str]
            Second set of nodes
        Z : Set[str]
            Conditioning set

        Returns
        -------
        bool
            True if X and Y are d-separated given Z
        """
        return nx.d_separated(self.graph, X, Y, Z)

    def visualize(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the causal graph.

        Parameters
        ----------
        output_path : str, optional
            Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph)
            plt.figure(figsize=(12, 8))

            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color='lightblue',
                node_size=3000,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray'
            )

            plt.title(f"Causal Graph: {self.name}")

            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()

        except ImportError:
            print("Matplotlib required for visualization")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dictionary representation.

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            'name': self.name,
            'nodes': [
                {'id': node, **self.node_metadata.get(node, {})}
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'strength': edge.strength,
                    'confidence': edge.confidence,
                    'mechanism': edge.mechanism
                }
                for edge in self.edges
            ]
        }

    def to_json(self, path: str) -> None:
        """
        Save graph to JSON file.

        Parameters
        ----------
        path : str
            Output file path
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalGraph':
        """
        Load graph from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation

        Returns
        -------
        CausalGraph
            Loaded graph
        """
        graph = cls(name=data['name'])

        # Add nodes
        for node_data in data['nodes']:
            graph.add_node(
                node_data['id'],
                node_type=node_data.get('type', 'variable'),
                metadata=node_data.get('metadata', {})
            )

        # Add edges
        for edge_data in data['edges']:
            graph.add_edge(
                edge_data['source'],
                edge_data['target'],
                strength=edge_data.get('strength', 1.0),
                confidence=edge_data.get('confidence', 1.0),
                mechanism=edge_data.get('mechanism', '')
            )

        return graph

    @classmethod
    def from_json(cls, path: str) -> 'CausalGraph':
        """
        Load graph from JSON file.

        Parameters
        ----------
        path : str
            Input file path

        Returns
        -------
        CausalGraph
            Loaded graph
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class StructuralCausalModel:
    """
    Structural Causal Model (SCM) with functional equations.

    An SCM defines how each variable is generated from its parents
    and exogenous noise.
    """

    def __init__(self, causal_graph: CausalGraph):
        """
        Initialize structural causal model.

        Parameters
        ----------
        causal_graph : CausalGraph
            Underlying causal graph
        """
        self.graph = causal_graph
        self.functions: Dict[str, Callable] = {}
        self.noise_distributions: Dict[str, Any] = {}

    def set_function(
        self,
        node: str,
        function: Callable,
        noise_dist: Optional[Any] = None
    ) -> None:
        """
        Set structural equation for a node.

        Parameters
        ----------
        node : str
            Node identifier
        function : callable
            Function that computes node value from parents
            Signature: f(parent_values, noise) -> value
        noise_dist : optional
            Noise distribution for this variable
        """
        self.functions[node] = function
        if noise_dist is not None:
            self.noise_distributions[node] = noise_dist

    def sample(
        self,
        n_samples: int = 1,
        interventions: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample from the structural causal model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        interventions : dict, optional
            Dictionary of interventions {node: value}

        Returns
        -------
        dict
            Dictionary of samples for each variable
        """
        samples = {node: np.zeros(n_samples) for node in self.graph.graph.nodes()}

        # Sample in topological order
        for node in self.graph.get_topological_order():
            # Check if this node is intervened upon
            if interventions and node in interventions:
                samples[node] = np.full(n_samples, interventions[node])
            else:
                # Get parent values
                parents = self.graph.get_parents(node)
                parent_values = {p: samples[p] for p in parents}

                # Sample noise
                if node in self.noise_distributions:
                    noise = self.noise_distributions[node].rvs(n_samples)
                else:
                    noise = np.zeros(n_samples)

                # Compute value using structural equation
                if node in self.functions:
                    samples[node] = self.functions[node](parent_values, noise)
                else:
                    # Default: just use noise
                    samples[node] = noise

        return samples

    def compute_counterfactual(
        self,
        observed: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute counterfactual: What would happen if we intervened?

        Parameters
        ----------
        observed : dict
            Observed values
        interventions : dict
            Interventions to apply

        Returns
        -------
        dict
            Counterfactual values
        """
        # This is a simplified version
        # Full counterfactual computation requires abduction-action-prediction

        # For now, we sample with interventions
        samples = self.sample(n_samples=1000, interventions=interventions)

        # Return means
        return {node: np.mean(values) for node, values in samples.items()}
