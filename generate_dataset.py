import argparse
import trimesh
import pathlib
import numpy as np
from tqdm import tqdm
import h5py


def load_off_file(filename: pathlib.Path) -> trimesh.Trimesh:
    """
    Load 3D model from OFF file and return mesh.
    """
    mesh = trimesh.load(filename)
    return mesh


def sample_point_cloud(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """
    Uniformly sample n_points from a mesh 3D model.
    """
    point_cloud = mesh.sample(n_points)
    return np.array(point_cloud)


def create_dataset(filename: str, modelnet_dir: str, n_points: int) -> None:
    """
    Sample point clouds and save dataset in a HDF5 file.
    """
    modelnet_dir = pathlib.Path(modelnet_dir)
    point_clouds = {"train": [], "test": []}
    labels = {"train": [], "test": []}
    class_names = []
    label = 0
    # Load all OFF files and save sampled point clouds with labels
    for class_dir in modelnet_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        class_names.append(class_name)
        print(f'Loading files for class "{class_name}" (label: {label})...')
        for suffix in point_clouds.keys():
            print(f"Loading {suffix} data...")
            files_dir = class_dir / suffix
            for off_filename in tqdm(sorted(files_dir.glob("*.off"))):
                mesh = load_off_file(off_filename)
                point_cloud = sample_point_cloud(mesh, n_points)
                point_clouds[suffix].append(point_cloud)
                labels[suffix].append(label)
        label += 1

    with h5py.File(filename, "w") as f:
        f.create_dataset("tr_cloud", data=point_clouds["train"])
        f.create_dataset("tr_labels", data=labels["train"])
        f.create_dataset("test_cloud", data=point_clouds["test"])
        f.create_dataset("test_labels", data=labels["test"])
        f.create_dataset("class_names", data=class_names)
    print(f"Point cloud dataset saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate point cloud dataset from ModelNet."
    )
    parser.add_argument("modelnet_dir", type=str, default="ModelNet40")
    parser.add_argument("output_dir", type=str, default="data")
    parser.add_argument("n_points", type=int, default=10000)
    args = parser.parse_args()
    modelnet_dir = args.modelnet_dir
    output_dir = args.output_dir
    n_points = args.n_points

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = output_dir + "/" + modelnet_dir.split("/")[-1] + "_cloud.h5"
    create_dataset(output_file, modelnet_dir, n_points)
