import torch
import torch.nn as nn
import numpy as np

from torch_geometric.data import Data

from ..data_utils import BaseDataset, subsample_indices


class DeepSetsDataset(BaseDataset):
    def __init__(
        self,
        point_clouds: np.ndarray,
        labels: np.ndarray,
        m: int,
        sample_method: str = "fps",
        point_cloud_transforms: list[callable] | None = None,
        **kwargs,
    ):
        """
        ModelNet classification dataset for Delaunay GNN.
        """
        super().__init__(
            point_clouds,
            labels,
            sample_method,
            point_cloud_transforms,
        )
        self.m = m

    def __getitem__(self, index: int) -> Data:
        P = self.X[index]
        P = self.apply_transforms(P)
        y = torch.tensor(self.y[index], dtype=torch.long)
        x_idx = subsample_indices(P, self.m, self.sample_method)
        x = torch.tensor(P[x_idx], dtype=torch.float)
        data = Data(x=x, y=y)
        return data
