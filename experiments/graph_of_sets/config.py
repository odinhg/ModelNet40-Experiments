from .model import GraphOfSetsModel 
from .dataset import GraphOfSetsDataset 

# Experiment config
model_name = "Graph of Sets"
model_constructor = GraphOfSetsModel 
dataset_constructor = GraphOfSetsDataset 
lr = 1e-3

# Values to use for m, k, use_edge_density
dataset_params = [
        (8, 128, False),
        (8, 128, True),
        (8, 256, False),
        (8, 256, True),
        (8, 512, False),
        (8, 512, True),
        (16, 64, False),
        (16, 64, True),
        (16, 128, False),
        (16, 128, True),
        (16, 256, False),
        (16, 256, True),
        (32, 32, False),
        (32, 32, True),
        (32, 64, False),
        (32, 64, True),
        (32, 128, False),
        (32, 128, True),
        (64, 16, False),
        (64, 16, True),
        (64, 32, False),
        (64, 32, True),
        (64, 64, False),
        (64, 64, True),
        (128, 8, False),
        (128, 8, True),
        (128, 16, False),
        (128, 16, True),
        (128, 32, False),
        (128, 32, True),
        ]

# Parameters for the model
model_params = {
    "in_dim": 3,
    "hidden_dim_1": 128,
    "hidden_dim_2": 466,
    "out_dim": 40,
    "n_layers_1": 2,
    "n_layers_2": 2,
    }
