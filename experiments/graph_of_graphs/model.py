import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..layers import GraphConvBlock, FCClassifier
from ..data_utils import GOGData, generate_batch_tensor


class GraphOfGraphsModel(nn.Module):
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
        Graph of Sets for ModelNet classification.
        """
        super().__init__()

        # Local Graph convolutional layer
        self.graphconv_1 = GraphConvBlock(in_dim, hidden_dim_1, n_layers_1, edge_dim=1, global_pool=None)

        # Global Graph convolutional layer
        self.graphconv_2 = GraphConvBlock(hidden_dim_1 + 3, hidden_dim_2, n_layers_2, edge_dim, global_pool)

        # Fully Connected Classifier
        self.classifier = FCClassifier(hidden_dim_2, out_dim)

    def forward(self, data: GOGData) -> torch.Tensor:
        # Local graphs
        out = self.graphconv_1(data)
        out = out.view(data.pos.shape[0], -1, out.shape[-1])
        out, _ = torch.max(out, dim=1, keepdim=False)

        # Global graph
        data.x = torch.cat((out, data.pos), dim=-1)
        data.edge_index = data.global_edge_index
        data.edge_attr = data.global_edge_attr
        data.batch = generate_batch_tensor(data.pos.size(0) // data.batch_size, data.batch_size).to(data.x.device)
        out = self.graphconv_2(data)

        out = self.classifier(out)
        return out
