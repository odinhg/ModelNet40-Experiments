import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..layers import Set2Vec, GraphConvBlock, FCClassifier


class GlobalGraphOfSetsModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        out_dim: int,
        n_layers_1: int,
        n_layers_2: int,
        edge_dim: int,
        global_pool: str = "max",
        **kwargs,
    ):
        """
        Global Graph of Sets for ModelNet classification.
        """
        super().__init__()

        # Set embedding layer
        self.setnn = Set2Vec(in_dim, hidden_dim_1, n_layers_1)

        # Graph convolutional layer
        self.graphconv = GraphConvBlock(hidden_dim_1, hidden_dim_2, n_layers_2, edge_dim, global_pool)

        # Fully Connected Classifier
        self.classifier = FCClassifier(hidden_dim_2, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        data.x = self.setnn(data.pos)
        out = self.graphconv(data)
        out = self.classifier(out)
        return out
