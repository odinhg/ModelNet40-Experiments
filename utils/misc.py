import numpy as np
import random
import torch
import torch.nn as nn
from pathlib import Path

def get_number_of_parameters(model: nn.Module, trainable_only: bool=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_random_seeds(seed: int=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_experiment_names(basepath: Path=Path("experiments")) -> list[str]:
    return [path.name for path in basepath.iterdir() if path.is_dir() and "__" not in path.name]
