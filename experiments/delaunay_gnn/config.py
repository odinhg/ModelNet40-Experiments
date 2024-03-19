from .model import DelaunayGNNModel
from .dataset import DelaunayGNNDataset

# Experiment config
model_name = "Delaunay GNN"
model_constructor = DelaunayGNNModel
dataset_constructor = DelaunayGNNDataset

# Values to use for m, k, use_edge_density
dataset_params = [
        (128, 0, False),
        (128, 0, True),
        (256, 0, False),
        (256, 0, True),
        (512, 0, False),
        (1024, 0, False),
        (2048, 0, False),
        ]

# Parameters for the model
model_params = {
    "in_dim": 3,
    "hidden_dim": 512,
    "out_dim": 40,
    "n_layers": 2,
    }
