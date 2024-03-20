from .model import SetOfSetsModel 
from .dataset import SetOfSetsDataset 

# Experiment config
model_name = "Set of Sets"
model_constructor = SetOfSetsModel 
dataset_constructor = SetOfSetsDataset 
lr = 1e-3

# Values to use for m, k, use_edge_density
dataset_params = [
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
    "hidden_dim": 512,
    "out_dim": 40,
    "n_layers": 2,
    }
