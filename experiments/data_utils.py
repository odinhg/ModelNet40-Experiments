import torch
import torch.nn.functional as F
import fpsample
import numpy as np
import scipy.spatial
from torch_geometric.utils import coalesce, to_undirected
from torch_geometric.data import Data


def subsample_indices(
    X: np.ndarray,
    n: int,
    method: str,
    replace: bool = False,
) -> np.ndarray:
    """
    Subsample n points from X using given method. Return indices in X.
    """
    assert method in ["fps", "uniform"]

    if method == "uniform":
        return np.random.choice(len(X), size=n, replace=replace)

    if method == "fps":
        return fpsample.bucket_fps_kdtree_sampling(X, n).astype(np.int64)


def build_delaunay_graph(X: np.ndarray) -> torch.Tensor:
    """
    Construct Delaunay graph from a point cloud X. Return edge indices.
    """
    triangulation = scipy.spatial.Delaunay(X, qhull_options="QJ")
    faces = torch.from_numpy(triangulation.simplices).t().contiguous().long()
    edge_index = torch.cat(
        [faces[:2], faces[::2], faces[::3], faces[1:3], faces[1::2], faces[2:4]], dim=1
    )  # 3-simplices to edges
    edge_index = coalesce(
        edge_index=edge_index, num_nodes=len(X)
    )  # Remove duplicate edges
    edge_index = to_undirected(edge_index=edge_index)
    return edge_index


def compute_distance_feature(
    D: np.ndarray, edge_index: torch.Tensor, p: float = 2.0
) -> torch.Tensor:
    """
    Compute distance based edge features from distance matrix D.
    """
    source_idx, target_idx = edge_index
    edge_distances = D[source_idx, target_idx]
    inv_distances = 1 / (edge_distances + 1) ** p
    edge_weight = inv_distances / inv_distances.max()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    return edge_weight


def compute_edge_density_feature(
    D: np.ndarray, edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute edge density values for edges. The argument D is the distance matrix from centroids to the larger point cloud. That is, D[i, j] = d(c_i, p_j).
    """
    # Faster numpy solution (not pretty)
    idx = np.argpartition(D, 2, axis=0)[:2]
    pairs, counts = np.unique(idx, axis=1, return_counts=True)
    count_dict = {tuple(pair): count for pair, count in zip(pairs.T, counts)}
    edge_densities = np.array(
        [
            count_dict.get(tuple(e), 0) + count_dict.get(tuple(np.flip(e)), 0)
            for e in edge_index.numpy().T
        ]
    )
    edge_weight = edge_densities / edge_densities.max()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    return edge_weight 

def sample_centroids(P: np.ndarray, m: int, sample_method: str="fps", replace: bool=False) -> tuple[np.ndarray]:
    # Sample centroids
    C_idx = subsample_indices(P, m, sample_method, replace)
    C = P[C_idx]
    return C, C_idx


def build_edge_features(D: np.ndarray, C_idx: np.ndarray, edge_index: torch.Tensor, use_edge_density: bool) -> torch.Tensor:
    """
    If use_edge_density is true, then D should be the distance matrix from C to P. Otherwise, D should be the pairwise distances in C. 
    """
    if use_edge_density:
        edge_distance_feature = compute_distance_feature(D[:, C_idx], edge_index)
        edge_density_feature = compute_edge_density_feature(D, edge_index)
        edge_attr = torch.stack([edge_distance_feature, edge_density_feature], dim=-1)
    else:
        edge_distance_feature = compute_distance_feature(D, edge_index)
        edge_attr = edge_distance_feature.unsqueeze(-1)
    return edge_attr


def sample_weighted_delaunay_graph(
    P: np.ndarray,
    m: int,
    sample_method: str = "fps",
    replace: bool = False,
    use_edge_density: bool = True,
):
    """
    Sample m points from P using farthest point sampling. Return node features (coordinates), edge index for the Delaunay graph, and distance and edge density edge features.
    """
    C, C_idx = sample_centroids(P, m, sample_method, replace)
    x = torch.tensor(C, dtype=torch.float)
    edge_index = build_delaunay_graph(C)

    if use_edge_density:
        D = scipy.spatial.distance.cdist(C, P)
        edge_distance_feature = compute_distance_feature(D[:, C_idx], edge_index)
        edge_density_feature = compute_edge_density_feature(D, edge_index)
        edge_attr = torch.stack([edge_distance_feature, edge_density_feature], dim=-1)
    else:
        D = scipy.spatial.distance.cdist(C, C)
        edge_distance_feature = compute_distance_feature(D, edge_index)
        edge_attr = edge_distance_feature.unsqueeze(-1)
    
    return x, edge_index, edge_attr


def sample_from_voronoi_cells(P: np.ndarray, D: np.ndarray, k: int) -> torch.Tensor:
    """
    Sample k points from P in each Voronoi cell corresponding to the centroids in C. Here, D is the distance matrix from C to P.
    """
    cell_idx = np.argmin(D, axis=0)
    samples = []
    for c in range(D.shape[0]):
        voronoi_cell = P[cell_idx == c]
        sample_idx = subsample_indices(voronoi_cell, k, method="uniform", replace=True)
        sample = P[sample_idx]
        samples.append(sample)
    x = torch.tensor(np.array(samples), dtype=torch.float)
    return x


def sample_set_of_sets(
    P: np.ndarray, m: int, k: int, sample_method: str = "fps"
) -> torch.Tensor:
    C, _ = sample_centroids(P, m, sample_method)
    D = scipy.spatial.distance.cdist(C, P)
    x = sample_from_voronoi_cells(P, D, k)
    return x


def sample_graph_of_sets(P: np.ndarray, m: int, k: int, use_edge_density: bool, sample_method: str="fps") -> tuple[torch.Tensor]:
    C, C_idx = sample_centroids(P, m, sample_method)
    pos = torch.tensor(C, dtype=torch.float)
    edge_index = build_delaunay_graph(C)

    D = scipy.spatial.distance.cdist(C, P)
    x = sample_from_voronoi_cells(P, D, k)

    if use_edge_density:
        edge_distance_feature = compute_distance_feature(D[:, C_idx], edge_index)
        edge_density_feature = compute_edge_density_feature(D, edge_index)
        edge_attr = torch.stack([edge_distance_feature, edge_density_feature], dim=-1)
    else:
        edge_distance_feature = compute_distance_feature(D[:, C_idx], edge_index)
        edge_attr = edge_distance_feature.unsqueeze(-1)

    return x, edge_index, edge_attr, pos


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        point_clouds: np.ndarray,
        labels: np.ndarray,
        sample_method: str = "fps",
        point_cloud_transforms: list[callable] | None = None,
    ):
        assert len(point_clouds) == len(
            labels
        ), f"Number of samples ({len(self.X)}) must be equal to the number of labels ({len(self.y)})."
        self.X = point_clouds
        self.y = labels
        self.length = len(self.y)
        self.sample_method = sample_method
        self.point_cloud_transforms = point_cloud_transforms

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Data:
        raise NotImplementedError

    def apply_transforms(self, P: np.ndarray) -> np.ndarray:
        if self.point_cloud_transforms:
            for transform in self.point_cloud_transforms:
                P = transform(P)
        return P
