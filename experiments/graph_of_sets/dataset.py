import torch
import torch.nn as nn
import numpy as np

from torch_geometric.data import Data

from ..data_utils import BaseDataset, sample_graph_of_sets


class GraphOfSetsDataset(BaseDataset):
    def __init__(
        self,
        point_clouds: np.ndarray,
        labels: np.ndarray,
        m: int,
        k: int,
        use_edge_density: bool,
        sample_method: str = "fps",
        point_cloud_transforms: list[callable] | None = None,
        **kwargs,
    ):
        """
        ModelNet classification dataset for Graph of Sets.
        """
        super().__init__(
            point_clouds,
            labels,
            sample_method,
            point_cloud_transforms,
        )
        self.m = m
        self.k = k
        self.use_edge_density = use_edge_density

    def __getitem__(self, index: int) -> Data:
        P = self.X[index]
        P = self.apply_transforms(P)
        y = torch.tensor(self.y[index], dtype=torch.long)
        x, edge_index, edge_attr, pos = sample_graph_of_sets(P, self.m, self.k, self.use_edge_density, self.sample_method)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=self.m, pos=pos)
        return data

