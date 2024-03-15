import torch
import torch.nn.functional as F
import fpsample
import numpy as np
import scipy.spatial
from torch_geometric.utils import coalesce, to_undirected

from time import time

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


def build_delaunay_graph(X: torch.Tensor) -> torch.Tensor:
    """
    Construct Delaunay graph from a point cloud X. Return edge indices.
    """
    points = X.cpu().numpy()
    triangulation = scipy.spatial.Delaunay(points, qhull_options="QJ")
    faces = torch.from_numpy(triangulation.simplices).t().contiguous()
    edge_index = torch.cat(
        [faces[:2], faces[::2], faces[::3], faces[1:3], faces[1::2], faces[2:4]], dim=1
    )  # 3-simplices to edges
    edge_index = coalesce(
        edge_index=edge_index, num_nodes=len(X)
    )  # Remove duplicate edges
    edge_index = to_undirected(edge_index=edge_index)
    return edge_index


def compute_distance_feature(
    D: torch.Tensor, edge_index: torch.Tensor, p: float = 2.0
) -> torch.Tensor:
    """
    Compute distance based edge features from distance matrix D.
    """
    source_idx, target_idx = edge_index
    edge_distances = D[source_idx, target_idx]
    edge_weight = 1 / (edge_distances + 1) ** p
    return edge_weight


def compute_edge_density_feature(
    D: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute edge density values for edges. The argument D is the distance matrix from centroids to the larger point cloud. That is, D[i, j] = d(c_i, p_j).
    """
    _, idx = torch.topk(D, k=2, dim=0, largest=False, sorted=False)
    idx = idx.numpy()
    count_close_pts = np.vectorize(lambda e: np.sum(np.all(idx.T == e, axis=1)))
    t0 = time()
    edge_densities = count_close_pts(edge_index.numpy().T)# + count_close_pts(edge_index.T.flip(0))
    t1 = time()
    print(f"{t1-t0:.5f}")
    edge_weight = edge_densities / edge_densities.max()#D.shape[0]
    print(edge_weight)
    return torch.tensor(edge_weight)
    """
    _, idx = torch.topk(D, k=2, dim=0, largest=False, sorted=False)
    count_close_pts = torch.vmap(lambda e: torch.sum(torch.all(idx.T == e, dim=1)))
    t0 = time()
    edge_densities = count_close_pts(edge_index.T)# + count_close_pts(edge_index.T.flip(0))
    t1 = time()
    print(f"{t1-t0:.5f}")
    edge_weight = edge_densities / edge_densities.max()#D.shape[0]
    return edge_weight
    """


def sample_weighted_delaunay_graph(
    P: np.ndarray,
    m: int,
    point_cloud_transforms: list[callable] | None = None,
    sample_method: str = "fps",
    replace: bool = False,
    use_edge_density: bool = True,
):
    # Apply transforms (if any)
    if point_cloud_transforms:
        for transform in point_cloud_transforms:
            P = transform(P)
    # Sample centroids
    C_idx = subsample_indices(P, m, sample_method, replace)
    C = torch.tensor(P[C_idx], dtype=torch.float)
    # Compute distances from centroids to all points
    D = torch.cdist(C, torch.tensor(P, dtype=torch.float))
    edge_index = build_delaunay_graph(C)

    edge_distance_feature = compute_distance_feature(D[:, C_idx], edge_index)
    if use_edge_density:
        edge_density_feature = compute_edge_density_feature(D, edge_index)
        edge_attr = torch.stack([edge_distance_feature, edge_density_feature], dim=-1)
    else:
        edge_attr = edge_distance_feature.unsqueeze(-1)


for _ in range(100):
    t0 = time()
    P = np.random.uniform(0, 1, (10000, 3))
    sample_weighted_delaunay_graph(P, 512)
    t1 = time()
    print(f"{t1-t0:.5f}")
