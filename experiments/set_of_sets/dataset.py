import torch
import torch.nn as nn
import numpy as np

from torch_geometric.data import Data

from ..data_utils import BaseDataset, sample_set_of_sets


class SetOfSetsDataset(BaseDataset):
    def __init__(
        self,
        point_clouds: np.ndarray,
        labels: np.ndarray,
        m: int,
        k: int,
        sample_method: str = "fps",
        point_cloud_transforms: list[callable] | None = None,
        **kwargs,
    ):
        """
        ModelNet classification dataset for Set of Sets. Samples m points using FPS. For each of these points, sample k points in its Voronoi cell.
        """
        super().__init__(
            point_clouds,
            labels,
            sample_method,
            point_cloud_transforms,
        )
        self.m = m
        self.k = k

    def __getitem__(self, index: int) -> Data:
        P = self.X[index]
        P = self.apply_transforms(P)
        y = torch.tensor(self.y[index], dtype=torch.long)
        x = sample_set_of_sets(P, self.m, self.k, self.sample_method)
        data = Data(x=x, y=y)
        return data
