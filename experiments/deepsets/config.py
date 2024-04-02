from .model import DeepSetsModel 
from .dataset import DeepSetsDataset

# Experiment config
model_name = "DeepSets"
model_constructor = DeepSetsModel 
dataset_constructor = DeepSetsDataset 
lr = 1e-3

# Values to use for m, k, use_edge_density
dataset_params = [
        (128, 0, False),
        (256, 0, False),
        (512, 0, False),
        (1024, 0, False),
        (2048, 0, False),
        (4096, 0, False),
        ]

# Parameters for the model
model_params = {
    "in_dim": 3,
    "hidden_dim": 512,
    "out_dim": 40,
    "n_layers": 3,
    }
