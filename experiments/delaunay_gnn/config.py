from .model import DelaunayGNNModel
from .dataset import DelaunayGNNDataset

# Experiment config
model_name = "Delaunay GNN"
model_constructor = DelaunayGNNModel
dataset_constructor = DelaunayGNNDataset

dataset_params = {
        "m": [128, 256, 512, 1024, 2048], 
        "k": [0], 
        "use_edge_density": [False],
        }

model_params = {
    "in_dim": 3,
    "hidden_dim": 512,
    "out_dim": 40,
    "n_layers": 2,
    }
