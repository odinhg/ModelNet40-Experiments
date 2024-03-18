import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm


def subtract_max(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Subtract maximum as done in DeepSets.
    """
    x_dense, mask = to_dense_batch(x, batch)
    x_max, _ = torch.max(x_dense, dim=-2, keepdim=True)
    x_dense = x_dense - x_max
    out = x_dense[mask]
    return out


class FCClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        p_dropout: float = 0.5,
    ):
        """
        Fully connected classifier head with 3 layers, ReLU, batch normalization and dropout.
        """
        super().__init__()

        # Fully Connected Classifier
        self.layers = nn.Sequential(
            Linear(in_dim, in_dim, weight_initializer="glorot"),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            Linear(in_dim, in_dim, weight_initializer="glorot"),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            Linear(in_dim, out_dim, weight_initializer="glorot"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out


class GraphConvBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
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
                GATv2Conv(in_dim, out_dim, edge_dim=edge_dim),
                "x, edge_index, edge_attr -> x",
            ),
            nn.ReLU(),
            (BatchNorm(out_dim), "x -> x"),
        ]

        for _ in range(n_layers - 1):
            layers += [
                (
                    GATv2Conv(out_dim, out_dim, edge_dim=edge_dim, share_weights=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (BatchNorm(out_dim), "x -> x"),
            ]

        if global_pool == "mean":
            layers.append((global_mean_pool, "x, batch -> x"))
        else:
            layers.append((global_max_pool, "x, batch -> x"))

        self.gnn_layers = Sequential("x, edge_index, edge_attr, batch", layers)

    def forward(self, data: Data) -> torch.Tensor:
        out = self.gnn_layers(data.x, data.edge_index, data.edge_attr, data.batch)
        return out


class Set2Vec(nn.Module):
    def __init__(
        self,
    ):
        """
        Set neural network as in DeepSets. Used by Set-of-Sets and Graph-of-Sets models.
        """
        super().__init__()
        # TODO: Re-implement PermiEqui max from DeepSets. Use subtract_max function.
        pass
