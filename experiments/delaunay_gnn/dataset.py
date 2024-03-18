import torch
import torch.nn as nn
import numpy as np

from torch_geometric.data import Data

from ..data_utils import BaseDataset, sample_weighted_delaunay_graph


class DelaunayGNNDataset(BaseDataset):
    def __init__(
        self,
        point_clouds: np.ndarray,
        labels: np.ndarray,
        m: int,
        k: int = 0,
        use_edge_density: bool = False,
        sample_method: str = "fps",
        point_cloud_transforms: list[callable] | None = None,
    ):
        """
        ModelNet classification dataset for Delaunay GNN.
        """
        super().__init__(
            point_clouds,
            labels,
            sample_method,
            use_edge_density,
            point_cloud_transforms,
        )
        self.m = m

    def __getitem__(self, index: int) -> Data:
        P = self.X[index]
        P = self.apply_transforms(P)
        y = torch.tensor(self.y[index], dtype=torch.long)
        x, edge_index, edge_attr = sample_weighted_delaunay_graph(
            P,
            self.m,
            sample_method=self.sample_method,
            use_edge_density=self.use_edge_density,
        )
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
