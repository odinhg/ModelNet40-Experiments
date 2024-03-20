import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..layers import GraphConvBlock, FCClassifier


class DelaunayGNNModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        edge_dim: int,
        global_pool: str = "max",
        **kwargs,
    ):
        """
        Delaunay GNN for ModelNet classification.
        """
        super().__init__()

        # Graph convolutional layers
        self.graphconv = GraphConvBlock(in_dim, hidden_dim, n_layers, edge_dim, global_pool)

        # Fully Connected Classifier
        self.classifier = FCClassifier(hidden_dim, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        out = self.graphconv(data)
        out = self.classifier(out)
        return out
