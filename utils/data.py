import h5py
import pathlib
import numpy as np
from tqdm import tqdm


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
