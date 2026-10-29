# Rock Bolt Identification

## Overview

This repository provides the implementation of an **automatic rock bolt identification** framework for 3D point clouds acquired in underground mining environments.

The workflow combines geometry-based filtering and deep learning-based semantic segmentation to identify rock-bolt points from complex underground point clouds.

The framework consists of two main stages:

1. **Stage 1 – Automated Filtering:** Geometric feature-based filtering using local eigenvalue analysis, surface curvature and spatial clustering to extract potential rock-bolt regions from the raw point cloud.
2. **Stage 2 – Semantic Segmentation:** Deep learning-based point-wise classification to distinguish rock-bolt points from non-bolt points.

The identified rock bolts are subsequently visualised on the original point cloud, and their coordinates are exported for further analysis.

**An example dataset and intermediate processing outputs are provided in this repository**, allowing users to execute the complete workflow or start directly from individual processing stages.

## Execution Instructions

### Step 1: Input Data Preparation

An example underground mine point cloud is provided in the `Raw Data` folder as a `.txt` file containing XYZ coordinates.

The corresponding manually identified ground-truth rock-bolt points are provided as individual `.txt` files in the `Ground Truth` folder.

These datasets serve as examples for testing the automated identification workflow.

### Step 2: Stage 1 – Automated Filtering

Run the MATLAB script:

`Stage1_Automated_filtering.m`

This stage processes the raw point cloud using local eigenvalue analysis to calculate geometric features, particularly surface curvature. Curvature-based filtering and DBSCAN clustering are subsequently applied to extract candidate rock-bolt regions while reducing irrelevant background points.

**Input:** `Raw Data/` – Raw point cloud (`.txt`)

**Output:** `Filtered Data/` – Filtered point cloud (`.txt`)

The filtered point cloud is automatically saved in the `Filtered Data` folder.

The output generated from the provided example dataset is already included in this folder. Therefore, users may proceed directly to the next step without rerunning Stage 1.

### Step 3: Convert Filtered Point Clouds to NumPy

Run the Python script:

`Bolt_Data_Converter_To_Numpy.py`

This script converts the curvature-filtered point cloud into a class-labelled NumPy array suitable for the semantic segmentation model.

Ground-truth rock-bolt points are assigned a label of `1`, while the remaining points are assigned a label of `0`.

Each point is represented by seven attributes:

| Column | Description                               |
| ------ | ----------------------------------------- |
| 1–3    | X, Y, Z coordinates                       |
| 4–6    | R, G, B values                            |
| 7      | Class label (1 = Rock Bolt, 0 = Non-Bolt) |

The generated `.npy` files are automatically saved to the required directories. This prepares the input data for the subsequent semantic segmentation stage without requiring manual file conversion or relocation.

### Step 4: Stage 2 – Semantic Segmentation

Navigate to the `Stage 2_Semantic Segmentation` directory and run:

`test_semseg_code.py`

This stage applies the trained semantic segmentation model to the curvature-filtered point cloud to classify individual points as rock-bolt or non-bolt points.

The model generates point-wise predictions, identifying regions corresponding to rock bolts within the filtered point cloud.

The prediction visualisation in OBJ format is stored in:

`Stage 2_Semantic Segmentation/log/sem_seg/pointnet2_sem_seg/visual/`

The point-wise prediction labels are used for subsequent visualisation in MATLAB.

**Pretrained Model and Optional Retraining**

The semantic segmentation model has been trained using multiple underground mining point-cloud datasets to support rock-bolt identification in a mine site.

Users can apply the existing trained model for inference on the provided example dataset or their own data.

Additionally, the model can be further trained using additional datasets by running:

`train_semseg.py`

For retraining, users must prepare their own point-cloud data and corresponding manually labelled ground-truth rock-bolt points, following the required input data structure.

This allows the model to be adapted to new mining **environments**, scanning conditions and rock-bolt configurations.

### Step 5: Visualise Identified Rock Bolts

Run the MATLAB script:

`Rock_Bolt_Identification_Visualiser.m`

This script reads the original raw point cloud, the curvature-filtered point cloud and the predicted point-wise classification labels.

Points classified as rock bolts (`Label = 1`) are extracted from the filtered point cloud and visualised over the original raw point cloud.

The resulting MATLAB figure displays:

- **Grey:** Original raw point cloud
- **Red:** Predicted rock-bolt points

The identified rock-bolt points are automatically exported as XYZ coordinates in `.txt` format to:

`Predicted Bolts/`

The predicted bolt-point files are provided for the example dataset, allowing users to inspect the final identification results without necessarily executing the preceding stages.

## Running Individual Stages

The repository includes the example input dataset and intermediate outputs for the processing stages.

Consequently, **users do not need to execute the complete workflow every time**.

Depending on their requirements, users can begin from:

- **Stage 1:** Process raw point clouds using geometric filtering.
- **Data Conversion:** Use the existing filtered point clouds to generate labelled NumPy files.
- **Stage 2:** Apply the trained semantic segmentation model to prepared point-cloud data.
- **Visualisation:** Inspect existing predictions and extract identified rock-bolt coordinates.

For new datasets, users should ensure that the input files, ground-truth data and prediction labels follow the expected formats and maintain consistent point ordering.

## Requirements

The workflow uses both MATLAB and Python.

**MATLAB**

- Computer Vision Toolbox
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox for parallel eigenvalue computation

**Python**

- NumPy
- tqdm
- PyTorch
- Associated dependencies supplied with the semantic segmentation framework

A compatible Python environment and the required deep learning dependencies must be configured before running the semantic segmentation scripts.

## Notes

- Input point clouds are expected to contain XYZ coordinates in space-separated `.txt` files.
- Ground-truth labels are generated through coordinate matching between the filtered point cloud and manually identified bolt points.
- The semantic segmentation predictions must retain the same point ordering as the corresponding filtered point cloud.
- The provided example outputs enable users to inspect and test individual stages independently.
- The final exported TXT files contain predicted rock-bolt points only, preserving their original XYZ coordinates.
