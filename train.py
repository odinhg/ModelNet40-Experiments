import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from experiments.delaunay_gnn import DelaunayGraphDataset, DelaunayGNN

from utils.data import load_and_split_dataset
from utils.misc import get_number_of_parameters
from utils.transforms import random_rotate_and_scale, normalize

# Experiment config
m = 1024 
use_edge_density = False 
dataset_filename = "data/ModelNet40_cloud.h5"
val_size = 0.10
train_transforms = [random_rotate_and_scale, normalize]
val_transforms = [normalize]
validate_interval = 10
validate_repeat = 5
lr = 1e-4
batch_size = 16 
epochs = 250
num_workers = 8 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = DelaunayGNN(
    in_dim=3,
    hidden_dim=512,
    out_dim=40,
    n_layers=2,
    edge_dim=2 if use_edge_density else 1,
).to(device)

print(f"# Trainable params: {get_number_of_parameters(model)}")

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=list(range(50, epochs, 50)), gamma=0.5
)

X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_and_split_dataset(
    dataset_filename, val_size
)

train_dataset = DelaunayGraphDataset(
    X_train, y_train, m=m, use_edge_density=use_edge_density, point_cloud_transforms=train_transforms
)
train_dl = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True,
)

val_dataset = DelaunayGraphDataset(
    X_val, y_val, m=m, use_edge_density=use_edge_density, point_cloud_transforms=train_transforms
)
val_dl = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=False,
)

for epoch in range(epochs):
    print(f"Epoch {epoch+1:03}/{epochs}")
    model.train()
    train_accuracies = []
    train_losses = []
    for data in (pbar := tqdm(train_dl)):
        data.to(device)
        labels = data.y
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        train_accuracy = (
            torch.mean((labels == torch.argmax(logits, dim=-1)).float())
            .detach()
            .cpu()
            .numpy()
        )
        train_accuracies.append(train_accuracy)
        train_losses.append(loss.item())
        pbar.set_description(
            f"Mean train loss: {np.mean(train_losses):.4f} | Mean train accuracy: {np.mean(train_accuracies):.4f}"
        )
    scheduler.step()
    
    # Validation step
    if (epoch + 1) % validate_interval == 0:
        model.eval()
        with torch.no_grad():
            val_accuracies = []
            for _ in range(validate_repeat):
                total_correct = 0
                total_count = 0
                for data in tqdm(val_dl):
                    data.to(device)
                    labels = data.y
                    logits = model(data)
                    correct = (
                        torch.sum((labels == torch.argmax(logits, dim=-1)).float())
                        .detach()
                        .cpu()
                    )
                    total_correct += correct
                    total_count += len(labels)
                val_accuracy = total_correct / total_count
                val_accuracies.append(val_accuracy)
            print(
                f"Validation Accuracy: {np.mean(val_accuracies):.4f} (±{np.std(val_accuracies):.4f})"
            )
        model.train()
