import wandb
import torch
import numpy as np
import random
from tqdm import tqdm
from itertools import product

from experiments.delaunay_gnn.config import *

from utils.data import load_dataset, create_dataloaders
from utils.misc import get_number_of_parameters
from utils.transforms import random_rotate_and_scale, normalize


# Global settings (applies to all experiments)
dataset_filename = "data/ModelNet40_cloud.h5"
val_size = 0.10
train_transforms = [random_rotate_and_scale, normalize]
val_transforms = [normalize]
validate_interval = 10
validate_repeat = 5
lr = 1e-3
batch_size = 16
epochs = 100
num_workers = 8
seed = 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset_dict = load_dataset(dataset_filename)

# Run experiments for all given combinations of hyper-parameters
print(f"Running experiments for {model_name}")
for m, k, use_edge_density in product(*dataset_params): 
    if (m, k, use_edge_density) in dataset_params_exclude:
        continue
    print(f"Using parameters: m = {m}, k = {k}, use_edge_density = {use_edge_density}")
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_dl, val_dl = create_dataloaders(
        data=dataset_dict,
        dataset_constructor=dataset_constructor,
        val_size=val_size,
        m=m,
        k=k,
        use_edge_density=use_edge_density,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        random_state=seed,
    )

    model = model_constructor(**model_params, edge_dim=2 if use_edge_density else 1).to(
        device
    )
    print(f"# Trainable params: {get_number_of_parameters(model)}")
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(20, epochs, 20)), gamma=0.5
    )

    # Start wandb run
    wandb.init(
        project="ModelNet40-Experiments",
        config={
            "initial_learning_rate": lr,
            "architecture": model_name,
            "dataset": dataset_filename,
            "epochs": epochs,
            "m": m,
            "k": k,
            "use_edge_density": use_edge_density,
            "val_size": val_size,
            "batch_size": batch_size,
            "validate_interval": validate_interval,
            "validate_repeat": validate_repeat,
            "train_transforms": ", ".join(
                [transform.__name__ for transform in train_transforms]
            ),
            "val_transforms": ", ".join(
                [transform.__name__ for transform in val_transforms]
            ),
        },
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
            wandb.log({"train_loss": loss.item(), "train_acc": train_accuracy})
        scheduler.step()
        wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

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
            wandb.log(
                {
                    "mean_val_acc": np.mean(val_accuracies),
                    "min_val_acc": np.min(val_accuracies),
                    "max_val_acc": np.max(val_accuracies),
                }
            )
            model.train()
    wandb.finish()
