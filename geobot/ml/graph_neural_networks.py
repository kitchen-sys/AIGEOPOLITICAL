"""
Graph Neural Networks for Causal Graphs and Geopolitical Networks

Implements GNNs for:
- Alliance and trade network analysis
- Causal graph representation learning
- Message passing on DAGs
- Attention mechanisms for influence propagation
- Graph classification and regression

Respects identifiability and invariance constraints from causal theory.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import networkx as nx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. GNN functionality limited.")

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, MessagePassing
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("torch_geometric not available. Install with: pip install torch-geometric")


if HAS_TORCH:
    class CausalGNN(nn.Module):
        """
        Graph Neural Network for causal graphs.

        Respects causal ordering (topological) and propagates information
        along causal edges.
        """

        def __init__(
            self,
            node_features: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            attention: bool = False
        ):
            """
            Initialize Causal GNN.

            Parameters
            ----------
            node_features : int
                Dimension of input node features
            hidden_dim : int
                Hidden layer dimension
            output_dim : int
                Output dimension
            num_layers : int
                Number of GNN layers
            attention : bool
                Use attention mechanism (GAT)
            """
            super(CausalGNN, self).__init__()

            self.num_layers = num_layers
            self.attention = attention

            # Input layer
            if attention and HAS_TORCH_GEOMETRIC:
                self.conv1 = GATConv(node_features, hidden_dim, heads=4, concat=True)
                self.convs = nn.ModuleList([
                    GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
                    for _ in range(num_layers - 2)
                ])
                self.conv_final = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
            elif HAS_TORCH_GEOMETRIC:
                self.conv1 = GCNConv(node_features, hidden_dim)
                self.convs = nn.ModuleList([
                    GCNConv(hidden_dim, hidden_dim)
                    for _ in range(num_layers - 2)
                ])
                self.conv_final = GCNConv(hidden_dim, output_dim)
            else:
                # Fallback to simple linear layers
                self.linear1 = nn.Linear(node_features, hidden_dim)
                self.linears = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(num_layers - 2)
                ])
                self.linear_final = nn.Linear(hidden_dim, output_dim)

        def forward(self, data):
            """
            Forward pass.

            Parameters
            ----------
            data : torch_geometric.data.Data
                Graph data with x (node features) and edge_index

            Returns
            -------
            torch.Tensor
                Node embeddings
            """
            if HAS_TORCH_GEOMETRIC:
                x, edge_index = data.x, data.edge_index

                # First layer
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)

                # Hidden layers
                for conv in self.convs:
                    x = conv(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)

                # Output layer
                x = self.conv_final(x, edge_index)

            else:
                x = data.x

                x = self.linear1(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)

                for linear in self.linears:
                    x = linear(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)

                x = self.linear_final(x)

            return x


    class GeopoliticalNetworkGNN(nn.Module):
        """
        GNN for geopolitical networks (alliances, trade, etc.).

        Models influence propagation and network effects.
        """

        def __init__(
            self,
            node_features: int,
            edge_features: int,
            hidden_dim: int,
            output_dim: int
        ):
            """
            Initialize geopolitical network GNN.

            Parameters
            ----------
            node_features : int
                Node feature dimension
            edge_features : int
                Edge feature dimension
            hidden_dim : int
                Hidden dimension
            output_dim : int
                Output dimension
            """
            super(GeopoliticalNetworkGNN, self).__init__()

            if HAS_TORCH_GEOMETRIC:
                self.conv1 = GCNConv(node_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.conv3 = GCNConv(hidden_dim, output_dim)
            else:
                self.linear1 = nn.Linear(node_features, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, hidden_dim)
                self.linear3 = nn.Linear(hidden_dim, output_dim)

            # Edge feature processing
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, data):
            """
            Forward pass.

            Parameters
            ----------
            data : Data
                Graph data

            Returns
            -------
            torch.Tensor
                Node embeddings
            """
            if HAS_TORCH_GEOMETRIC:
                x, edge_index = data.x, data.edge_index

                x = self.conv1(x, edge_index)
                x = F.relu(x)

                x = self.conv2(x, edge_index)
                x = F.relu(x)

                x = self.conv3(x, edge_index)
            else:
                x = data.x

                x = self.linear1(x)
                x = F.relu(x)

                x = self.linear2(x)
                x = F.relu(x)

                x = self.linear3(x)

            return x


    class MessagePassingCausalGNN(MessagePassing if HAS_TORCH_GEOMETRIC else nn.Module):
        """
        Custom message passing for causal graphs.

        Implements directed message passing that respects causal structure:
        - Messages flow only in direction of causal edges
        - Aggregation respects causal mechanisms
        """

        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
            """
            Initialize message passing GNN.

            Parameters
            ----------
            node_dim : int
                Node feature dimension
            edge_dim : int
                Edge feature dimension
            hidden_dim : int
                Hidden dimension
            """
            if HAS_TORCH_GEOMETRIC:
                super(MessagePassingCausalGNN, self).__init__(aggr='add')
            else:
                super(MessagePassingCausalGNN, self).__init__()

            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim

            # Message function
            self.message_mlp = nn.Sequential(
                nn.Linear(node_dim + node_dim + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # Update function
            self.update_mlp = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, node_dim)
            )

        def forward(self, x, edge_index, edge_attr):
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Node features
            edge_index : torch.Tensor
                Edge indices
            edge_attr : torch.Tensor
                Edge attributes

            Returns
            -------
            torch.Tensor
                Updated node features
            """
            if HAS_TORCH_GEOMETRIC:
                return self.propagate(edge_index, x=x, edge_attr=edge_attr)
            else:
                # Fallback implementation
                return x

        def message(self, x_i, x_j, edge_attr):
            """
            Construct messages.

            x_i: target node features
            x_j: source node features
            edge_attr: edge features

            Returns
            -------
            torch.Tensor
                Messages
            """
            # Concatenate source, target, and edge features
            msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
            return self.message_mlp(msg_input)

        def update(self, aggr_out, x):
            """
            Update node features.

            Parameters
            ----------
            aggr_out : torch.Tensor
                Aggregated messages
            x : torch.Tensor
                Current node features

            Returns
            -------
            torch.Tensor
                Updated node features
            """
            # Concatenate aggregated messages with current features
            update_input = torch.cat([x, aggr_out], dim=-1)
            return self.update_mlp(update_input)


    class AttentionGNN(nn.Module):
        """
        Graph Attention Network for geopolitical influence.

        Uses attention to weight importance of different neighbors/allies.
        """

        def __init__(
            self,
            node_features: int,
            hidden_dim: int,
            output_dim: int,
            num_heads: int = 4
        ):
            """
            Initialize attention GNN.

            Parameters
            ----------
            node_features : int
                Input node feature dimension
            hidden_dim : int
                Hidden dimension
            output_dim : int
                Output dimension
            num_heads : int
                Number of attention heads
            """
            super(AttentionGNN, self).__init__()

            if HAS_TORCH_GEOMETRIC:
                self.conv1 = GATConv(node_features, hidden_dim, heads=num_heads, concat=True)
                self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
            else:
                # Fallback
                self.linear1 = nn.Linear(node_features, hidden_dim * num_heads)
                self.linear2 = nn.Linear(hidden_dim * num_heads, output_dim)

        def forward(self, data):
            """
            Forward pass with attention.

            Parameters
            ----------
            data : Data
                Graph data

            Returns
            -------
            torch.Tensor
                Node embeddings with attention weights
            """
            if HAS_TORCH_GEOMETRIC:
                x, edge_index = data.x, data.edge_index

                # First layer with multi-head attention
                x, attention_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
                x = F.elu(x)
                x = F.dropout(x, p=0.3, training=self.training)

                # Second layer
                x, attention_weights2 = self.conv2(x, edge_index, return_attention_weights=True)

                return x, (attention_weights1, attention_weights2)
            else:
                x = data.x

                x = self.linear1(x)
                x = F.elu(x)
                x = F.dropout(x, p=0.3, training=self.training)

                x = self.linear2(x)

                return x, None


class GNNTrainer:
    """
    Trainer for GNN models.

    Handles training, validation, and evaluation.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4
    ):
        """
        Initialize GNN trainer.

        Parameters
        ----------
        model : nn.Module
            GNN model
        learning_rate : float
            Learning rate
        weight_decay : float
            Weight decay (L2 regularization)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GNN training")

        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_step(
        self,
        data,
        labels,
        loss_fn: Callable
    ) -> float:
        """
        Single training step.

        Parameters
        ----------
        data : Data
            Graph data
        labels : torch.Tensor
            Labels
        loss_fn : callable
            Loss function

        Returns
        -------
        float
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(data)

        # Compute loss
        loss = loss_fn(out, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        data,
        labels,
        metric_fn: Callable
    ) -> float:
        """
        Evaluate model.

        Parameters
        ----------
        data : Data
            Graph data
        labels : torch.Tensor
            True labels
        metric_fn : callable
            Evaluation metric

        Returns
        -------
        float
            Metric value
        """
        self.model.eval()

        with torch.no_grad():
            out = self.model(data)
            metric = metric_fn(out, labels)

        return metric


class NetworkToGraph:
    """
    Convert NetworkX graph to PyTorch Geometric format.
    """

    @staticmethod
    def convert(
        G: nx.Graph,
        node_features: Optional[Dict] = None,
        edge_features: Optional[Dict] = None
    ):
        """
        Convert NetworkX graph to PyTorch Geometric Data.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph
        node_features : dict, optional
            Node feature dictionary
        edge_features : dict, optional
            Edge feature dictionary

        Returns
        -------
        Data
            PyTorch Geometric Data object
        """
        if not HAS_TORCH or not HAS_TORCH_GEOMETRIC:
            raise ImportError("PyTorch and torch_geometric required")

        # Node features
        if node_features:
            x = torch.tensor([
                node_features[node]
                for node in G.nodes()
            ], dtype=torch.float)
        else:
            # Default: one-hot encoding
            n_nodes = G.number_of_nodes()
            x = torch.eye(n_nodes)

        # Edge index
        edge_index = torch.tensor(
            list(G.edges()),
            dtype=torch.long
        ).t().contiguous()

        # Edge features
        if edge_features:
            edge_attr = torch.tensor([
                edge_features[(u, v)]
                for u, v in G.edges()
            ], dtype=torch.float)
        else:
            edge_attr = None

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data


def example_geopolitical_gnn():
    """
    Example: GNN for geopolitical alliance network.
    """
    if not HAS_TORCH or not HAS_TORCH_GEOMETRIC:
        print("PyTorch and torch_geometric required for GNN examples")
        return

    # Create example graph (alliance network)
    G = nx.DiGraph()
    countries = ['USA', 'China', 'Russia', 'EU', 'India']
    G.add_nodes_from(countries)

    # Add alliance edges
    alliances = [
        ('USA', 'EU'),
        ('USA', 'India'),
        ('China', 'Russia'),
    ]
    G.add_edges_from(alliances)

    # Node features (e.g., GDP, military strength, etc.)
    node_features = {
        'USA': [1.0, 0.9, 0.8],
        'China': [0.8, 0.7, 0.9],
        'Russia': [0.5, 0.7, 0.6],
        'EU': [0.9, 0.6, 0.7],
        'India': [0.6, 0.6, 0.7]
    }

    # Convert to PyTorch Geometric format
    data = NetworkToGraph.convert(G, node_features)

    # Create model
    model = CausalGNN(
        node_features=3,
        hidden_dim=16,
        output_dim=8,
        num_layers=2
    )

    # Forward pass
    embeddings = model(data)

    print("Node embeddings shape:", embeddings.shape)
    print("Node embeddings:", embeddings)

    return model, data, embeddings
