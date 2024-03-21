import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..layers import Set2Vec, FCClassifier


class SetOfSetsModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        out_dim: int,
        n_layers_1: int,
        n_layers_2: int,
        **kwargs,
    ):
        """
        Set of Sets for ModelNet classification.
        """
        super().__init__()

        # Set embedding layers
        self.setnn_1 = Set2Vec(in_dim, hidden_dim_1, n_layers_1)
        self.setnn_2 = Set2Vec(hidden_dim_1, hidden_dim_2, n_layers_2)

        # Fully Connected Classifier
        self.classifier = FCClassifier(hidden_dim_2, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        out = self.setnn_1(data.x)
        out = out.view(data.batch_size, -1, out.shape[-1])
        out = self.setnn_2(out)
        out = self.classifier(out)
        return out
