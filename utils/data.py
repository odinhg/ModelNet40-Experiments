import h5py
import pathlib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


def load_dataset(filename: pathlib.Path | str) -> dict[np.ndarray]:
    """
    Load dataset from HDF5 file and return as dictionary.
    """
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    assert filename.is_file(), f'Supplied datafile "{filename}" does not exist.'
    print(f"Loading data from {filename}")
    with h5py.File(filename, "r") as f:
        data = {key: f[key][:] for key in tqdm(f.keys())}
    return data


def load_and_split_dataset(
    data: dict, val_size: float, random_state: int = 0
):
    X_train, y_train = data["tr_cloud"], data["tr_labels"]
    X_test, y_test = data["test_cloud"], data["test_labels"]
    class_names = data["class_names"]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def create_dataloaders(
    data: dict, 
    dataset_constructor: callable,
    val_size: float,
    m: int,
    k: int,
    use_edge_density: bool,
    batch_size: int,
    num_workers: int,
    train_transforms: list[callable],
    val_transforms: list[callable],
    random_state: int=0,
) -> tuple[DataLoader]:
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class_names,
    ) = load_and_split_dataset(data, val_size, random_state=random_state)

    train_dataset = dataset_constructor(
        X_train,
        y_train,
        m=m,
        k=k,
        use_edge_density=use_edge_density,
        point_cloud_transforms=train_transforms,
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = dataset_constructor(
        X_val,
        y_val,
        m=m,
        k=k,
        use_edge_density=use_edge_density,
        point_cloud_transforms=val_transforms,
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return train_dl, val_dl
