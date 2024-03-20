import torch
from utils.transforms import random_rotate_and_scale, normalize

# Global settings (applies to all experiments)
dataset_filename = "data/ModelNet40_cloud.h5"
val_size = 0.10
train_transforms = [random_rotate_and_scale, normalize]
val_transforms = [normalize]
validate_interval = 10
validate_repeat = 5
batch_size = 16
epochs = 100
num_workers = 8
seed = 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"
