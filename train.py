import wandb
import torch
import numpy as np
import random
import argparse
import importlib
from tqdm import tqdm

from utils.data import load_dataset, create_dataloaders
from utils.misc import get_number_of_parameters, set_random_seeds, get_experiment_names

# Import global configuration common for all experiments
import experiments.global_config as global_config

# Parse arguments and import experiment configuration
parser = argparse.ArgumentParser(description="Run ModelNet40 experiments for a selected architecture.")
parser.add_argument("experiment", help="name of experiment/architecture", choices=get_experiment_names())
parser.add_argument('--nolog', help="disable logging to wandb", action='store_true')
args = parser.parse_args()
config = importlib.import_module(".config", f"experiments.{args.experiment}")
use_wandb = not args.nolog

# Load dataset
dataset_dict = load_dataset(global_config.dataset_filename)

# Run experiments
print(f"Running experiments for {config.model_name}")
for m, k, use_edge_density in config.dataset_params:
    print(f"Using parameters: m = {m}, k = {k}, use_edge_density = {use_edge_density}")

    set_random_seeds(global_config.seed)

    train_dl, val_dl = create_dataloaders(
        data=dataset_dict,
        dataset_constructor=config.dataset_constructor,
        val_size=global_config.val_size,
        m=m,
        k=k,
        use_edge_density=use_edge_density,
        batch_size=global_config.batch_size,
        num_workers=global_config.num_workers,
        train_transforms=global_config.train_transforms,
        val_transforms=global_config.val_transforms,
        random_state=global_config.seed,
    )

    model = config.model_constructor(
        **config.model_params, edge_dim=2 if use_edge_density else 1
    ).to(global_config.device)
    no_parameters = get_number_of_parameters(model)
    print(f"# Trainable params: {no_parameters}")
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(20, global_config.epochs, 20)), gamma=0.5
    )

    if use_wandb:
        # Start wandb run
        wandb.init(
            project="ModelNet40-Experiments",
            config={
                "initial_learning_rate": config.lr,
                "architecture": config.model_name,
                "dataset": global_config.dataset_filename,
                "epochs": global_config.epochs,
                "m": m,
                "k": k,
                "use_edge_density": use_edge_density,
                "val_size": global_config.val_size,
                "batch_size": global_config.batch_size,
                "validate_interval": global_config.validate_interval,
                "validate_repeat": global_config.validate_repeat,
                "train_transforms": ", ".join(
                    [transform.__name__ for transform in global_config.train_transforms]
                ),
                "val_transforms": ", ".join(
                    [transform.__name__ for transform in global_config.val_transforms]
                ),
                "#parameters": no_parameters,
            },
        )

    for epoch in range(global_config.epochs):
        print(f"Epoch {epoch+1:03}/{global_config.epochs}")
        model.train()
        train_accuracies = []
        train_losses = []
        for data in (pbar := tqdm(train_dl)):
            data.to(global_config.device)
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
            if use_wandb:
                wandb.log({"train_loss": loss.item(), "train_acc": train_accuracy})
        scheduler.step()
        if use_wandb:
            wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

        # Validation step
        if (epoch + 1) % global_config.validate_interval == 0:
            model.eval()
            with torch.no_grad():
                val_accuracies = []
                for _ in range(global_config.validate_repeat):
                    total_correct = 0
                    total_count = 0
                    for data in tqdm(val_dl):
                        data.to(global_config.device)
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
            if use_wandb:
                wandb.log(
                    {
                        "mean_val_acc": np.mean(val_accuracies),
                        "min_val_acc": np.min(val_accuracies),
                        "max_val_acc": np.max(val_accuracies),
                    }
                )
            model.train()
    if use_wandb:
        wandb.finish()
