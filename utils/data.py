import h5py
import pathlib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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
    filename: pathlib.Path | str, val_size: float, random_state: int = 0
):
    data = load_dataset(filename)
    X_train, y_train = data["tr_cloud"], data["tr_labels"]
    X_test, y_test = data["test_cloud"], data["test_labels"]
    class_names = data["class_names"]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names
