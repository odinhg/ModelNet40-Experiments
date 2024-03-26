import torch
import torch.nn as nn
import numpy as np

from ..data_utils import BaseDataset, GOGData, sample_graph_of_graphs


class GraphOfGraphsDataset(BaseDataset):
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
        ModelNet classification dataset for Graph of Graphs.
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

    def __getitem__(self, index: int) -> GOGData:
        P = self.X[index]
        P = self.apply_transforms(P)
        y = torch.tensor(self.y[index], dtype=torch.long)

        x, edge_index, edge_attr, global_edge_index, global_edge_attr, pos = sample_graph_of_graphs(P, self.m, self.k, self.use_edge_density, self.sample_method)
        data = GOGData(x=x, edge_index=edge_index, edge_attr=edge_attr, global_edge_index=global_edge_index, global_edge_attr=global_edge_attr, y=y, pos=pos)
        
        return data

