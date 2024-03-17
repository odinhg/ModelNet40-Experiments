import numpy as np


def rotate_z(P: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate point cloud around z axis by theta (radians).
    """
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return P @ R.T


def random_rotate_and_scale(P: np.ndarray) -> np.ndarray:
    """
    Random z-rotation and scaling (same as DeepSets).
    """
    scale = np.random.rand(1, 3) * 0.45 + 0.8
    theta = np.random.uniform(-0.1, 0.1) * np.pi
    return rotate_z(P, theta) * scale


def standardize(P: np.ndarray) -> np.ndarray:
    """
    Standardize point cloud P.
    """
    mean = np.mean(P, axis=(0, 1), keepdims=True)
    std = np.std(P, axis=(0, 1), keepdims=True)
    return (P - mean) / std


def normalize(P: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud as done in PointNet++ implementation.
    """
    mean = np.mean(P, axis=0)
    m = np.max(np.sqrt(np.sum(P**2, axis=1)))
    return (P - mean) / m
