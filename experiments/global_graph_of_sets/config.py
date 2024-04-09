from .model import GlobalGraphOfSetsModel 
from .dataset import GlobalGraphOfSetsDataset 

# Experiment config
model_name = "Global Graph of Sets"
model_constructor = GlobalGraphOfSetsModel 
dataset_constructor = GlobalGraphOfSetsDataset 
lr = 1e-3

# Values to use for m, k, use_edge_density
dataset_params = [
        (16, 256, False),
        (8, 128, False),
        (8, 256, False),
        (8, 512, False),
        (16, 64, False),
        (16, 128, False),
        (16, 256, False),
        (32, 32, False),
        (32, 64, False),
        (32, 128, False),
        (64, 16, False),
        (64, 32, False),
        (64, 64, False),
        (128, 8, False),
        (128, 16, False),
        (128, 32, False),
        ]

# Parameters for the model
model_params = {
    "in_dim": 3,
    "hidden_dim_1": 500,
    "hidden_dim_2": 200,
    "out_dim": 40,
    "n_layers_1": 3,
    "n_layers_2": 2,
    }
