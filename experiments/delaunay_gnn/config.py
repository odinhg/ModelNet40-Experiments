from .model import DelaunayGNNModel
from .dataset import DelaunayGNNDataset

# Experiment config
model_name = "Delaunay GNN"
model_constructor = DelaunayGNNModel
dataset_constructor = DelaunayGNNDataset

# Values to use for m, k, use_edge_density
dataset_params = [
        [128, 256, 512, 1024, 2048], 
        [0], 
        [False, True],
        ]

# Combinations to exclude
dataset_params_exclude = [
        (512, 0, True),
        (1024, 0, True),
        (2048, 0, True),
        ]

# Parameters for the model
model_params = {
    "in_dim": 3,
    "hidden_dim": 512,
    "out_dim": 40,
    "n_layers": 2,
    }
