import torch.nn as nn

def get_number_of_parameters(model: nn.Module, trainable_only: bool=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
