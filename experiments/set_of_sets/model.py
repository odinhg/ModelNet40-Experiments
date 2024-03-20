import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..layers import Set2Vec, FCClassifier


class SetOfSetsModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers_1: int,
        n_layers_2: int,
        **kwargs,
    ):
        """
        Delaunay GNN for ModelNet classification.
        """
        super().__init__()

        # Graph convolutional layers
        self.setnn_1 = Set2Vec(in_dim, hidden_dim, n_layers_1)
        self.setnn_2 = Set2Vec(hidden_dim, hidden_dim, n_layers_2)

        # Fully Connected Classifier
        self.classifier = FCClassifier(hidden_dim, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        out = self.graphconv(data)
        out = self.classifier(out)
        return out
