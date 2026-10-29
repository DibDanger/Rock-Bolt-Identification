
import numpy as np
from pathlib import Path
from tqdm import tqdm

# %% 1. Define Input and Output Folders

# Relative paths based on Python script location
base_path = Path(__file__).resolve().parent

input_folder = base_path / "Filtered Data"
ground_truth_folder = base_path / "Ground Truth"

# Two output locations
output_folder_1 = (
    base_path / "Semantic Segmentation" / "data" / "rockbolt_data"
)

output_folder_2 = (
    base_path / "Semantic Segmentation" / "data" / "s3dis" / "rockbolt_data"
)

# Create output folders if they do not exist
output_folder_1.mkdir(parents=True, exist_ok=True)
output_folder_2.mkdir(parents=True, exist_ok=True)


# %% 2. Read Point Cloud

def read_point_cloud(file_path):

    points = []

    with open(file_path, "r") as file:
        for line in file:

            coords = line.strip().split()

            if len(coords) < 3:
                continue

            point = [float(coord) for coord in coords[:3]]

            # Add RGB values (white)
            point.extend([255, 255, 255])

            points.append(point)

    return points


# %% 3. Add Ground Truth Bolt Labels

def add_bolt_labels(bolt_folder, point_cloud):

    # Create lookup dictionary for XYZ coordinates
    point_lookup = {}

    for idx, point in enumerate(point_cloud):

        xyz = tuple(point[:3])

        point_lookup.setdefault(xyz, []).append(idx)

    # Read all ground truth bolt files
    ground_truth_files = sorted(bolt_folder.glob("*.txt"))

    for bolt_file in ground_truth_files:

        with open(bolt_file, "r") as file:

            for line in file:

                coords = line.strip().split()

                if len(coords) < 3:
                    continue

                bolt_point = tuple(float(x) for x in coords[:3])

                # Assign label 1 to matching bolt points
                if bolt_point in point_lookup:

                    for idx in point_lookup[bolt_point]:
                        point_cloud[idx].append(1)

                    # Prevent duplicate labelling
                    del point_lookup[bolt_point]

    return point_cloud


# %% 4. Add Non-Bolt Labels

def add_nonbolt_labels(point_cloud):

    for point in point_cloud:

        if len(point) == 6:
            point.append(0)

    return point_cloud


# %% 5. Process All Point Clouds

input_files = sorted(input_folder.glob("*.txt"))

print(f"Total point clouds found: {len(input_files)}")

for input_file in tqdm(input_files, desc="Processing Point Clouds"):

    file_name = input_file.stem

    # Read filtered point cloud
    point_cloud = read_point_cloud(input_file)

    # Corresponding ground truth folder
    bolt_folder = ground_truth_folder / file_name

    if bolt_folder.is_dir():

        point_cloud = add_bolt_labels(bolt_folder, point_cloud)

    else:

        print(f"Warning: No ground truth folder for {file_name}")
        print("Skipping this file to avoid incorrect labelling.")
        continue

    # Assign label 0 to remaining points
    point_cloud = add_nonbolt_labels(point_cloud)

    # Convert to NumPy array
    point_cloud = np.array(point_cloud, dtype=np.float64)

    # Save NPY file to both output locations
    output_file_1 = output_folder_1 / f"{file_name}.npy"
    output_file_2 = output_folder_2 / f"{file_name}.npy"

    np.save(output_file_1, point_cloud)
    np.save(output_file_2, point_cloud)

    # Display point counts
    bolt_count = np.sum(point_cloud[:, 6] == 1)
    nonbolt_count = np.sum(point_cloud[:, 6] == 0)

    print(
        f"{file_name}: "
        f"{bolt_count} bolt points, "
        f"{nonbolt_count} non-bolt points"
    )

print("\nAll available point clouds processed.")
