import torch
import torch.nn as nn


class FCClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()

        # Fully Connected Classifier
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out
