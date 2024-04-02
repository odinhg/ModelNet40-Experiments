import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..layers import Set2Vec


class DeepSetsModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        **kwargs,
    ):
        """
        DeepSets model for ModelNet classification.
        """
        super().__init__()

        # Set NN
        self.setnn = Set2Vec(in_dim, hidden_dim, n_layers)

        # Fully Connected Classifier
        self.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Dropout(p=0.5),
                                nn.Linear(hidden_dim, out_dim),
                                )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.view(data.batch_size, -1, data.x.shape[-1])
        out = self.setnn(x)
        out = self.classifier(out)
        return out
