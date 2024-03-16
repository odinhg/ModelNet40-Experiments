import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GENConv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm


class DelaunayGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        edge_dim: int,
        global_pool: str = "max",
    ):
        super().__init__()

        assert n_layers > 0
        assert global_pool in ["mean", "max"]

        layers = [
            (
                GENConv(in_dim, hidden_dim, aggr="max", num_layers=1, expansion=1, edge_dim=edge_dim),
                "x, edge_index, edge_attr -> x",
            ),
            nn.ReLU(),
            (BatchNorm(hidden_dim), "x -> x"),
        ]

        for _ in range(n_layers - 1):
            layers += [
                (
                    GENConv(hidden_dim, hidden_dim, aggr="max", num_layers=1, expansion=1, edge_dim=edge_dim),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (BatchNorm(hidden_dim), "x -> x"),
            ]

        if global_pool == "mean":
            layers.append((global_mean_pool, "x, batch -> x"))
        else:
            layers.append((global_max_pool, "x, batch -> x"))

        self.gnn_layers = Sequential("x, edge_index, edge_attr, batch", layers)

        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        out = self.gnn_layers(data.x, data.edge_index, data.edge_attr, data.batch)
        out = self.classifier(out)
        return out
