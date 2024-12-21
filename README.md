# 3D Reconstruction and Nearest 6-DoF Pose Estimation using Truncated Signed Distance Functions

## Overview

This repository contains a comprehensive pipeline for real-time **3D reconstruction** and **nearest 6 Degrees of Freedom (6DOF) pose estimation** from monocular RGB-D video data. The approach integrates various computer vision and robotics techniques to deliver accurate pose estimation, which is essential for applications in augmented reality, robotics, and autonomous systems.

- Real-time 3D model construction using **Truncated Signed Distance Function (TSDF)** fusion.
- Nearest estimation of the object’s **6DOF pose**.
- High-resolution 3D mesh generation using **Marching Cubes** and **Occupancy Dual Contouring** algorithms.

## Pipeline Overview

The pipeline consists of three main steps:

1. **3D Model Construction**: Using TSDF fusion from consecutive depth maps to create a dense 3D model.
2. **Pose Estimation**: Estimating the object's 6DOF pose based on the constructed model.
3. **Mesh Generation**: Generating a high-resolution 3D mesh for refined pose estimation through differential rendering.



## Data Preparation

### Expected Data Format

The data should be organized in the following structure:

```
data/
  kitchen-7scenes/
    camera-intrinsics.txt
    frame-000000.color.jpg
    frame-000000.depth.png
    frame-000000.pose.txt
    ...
```

- **camera-intrinsics.txt**: Camera intrinsic parameters.
- **frame-XXXXXX.color.jpg**: RGB images.
- **frame-XXXXXX.depth.png**: Depth maps.
- **frame-XXXXXX.pose.txt**: Camera poses.

### Data Sources

1. **Download the 7-Scenes Dataset** and place it in the data directory.
2. **Generate synthetic data** using Blender or Gazebo and organize it similarly.

## Configuration and Hyperparameters

The main script and notebooks allow configuring various parameters:

- **Volume Bounds**: Define the spatial bounds of the scene.
- **Voxel Size**: Determines the resolution of the volumetric grid.
- **Truncation Distance**: Controls the truncation of the signed distance function.

Example configuration:

```py
tsdf_voxel_size = 0.02  # 2cm
tsdf_trunc_margin = 0.1  # 10cm
```

## Output Explanation

The pipeline generates several outputs:

- **Meshes**: High-resolution 3D meshes saved in `.ply` format.
- **Point Clouds**: Point clouds saved in `.ply` format.

Output directories:
- bunny_outputs
- suzanne_outputs
- textured_scene_outputs

## Results

### 3D Reconstruction

The pipeline demonstrates 3D reconstruction using RGB-D images and camera poses, generating reconstructed meshes and point clouds.

### Textured Mesh Reconstruction

The concept of storing and updating TSDF values can be expanded to include updating the color value at each voxel, resulting in colored reconstructed meshes.

### Comparison of TSDF Hyper-parameters

The hyper-parameters in TSDF construction, such as voxel size and truncation distance, significantly impact the quality of reconstructed meshes.

### TSDF to Mesh Methods

Comparison between Marching Cubes and Occupancy Dual Contouring shows that Marching Cubes generates denser and noisier surfaces, while Occupancy Dual Contouring produces cleaner, lower-polygon meshes.

### Scene Reconstruction

The process can be applied to scenes where different views capture various parts of the environment, integrating all information into the TSDF.

### Performance Analysis

The GPU implementation achieves significant speedup compared to the CPU version, showcasing the advantages of using GPUs for computationally intensive tasks.
