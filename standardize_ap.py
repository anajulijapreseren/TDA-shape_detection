"""
This script includes a function 'scale_point_cloud' that processes pre-generated 3D point clouds of various shapes. 
The function scales and translates each point cloud such that:

1. The point cloud fits within a bounding box with an edge length of 'target_size'.
2. The center of the point cloud is centered around the origin (0, 0, 0) in the coordinate space.

This scaling ensures uniformity in size across all shapes, facilitating comparisons and analysis, 
while the translation aligns the centroid of each shape with the origin, standardizing their position.

When run as a script, it applies this function to a collection of point clouds (loaded from 'shapes_data.pkl'), 
retaining the original indices and labels. The processed point clouds are then saved in a new file 
('scaled_shapes_data.pkl'), preserving the order and classification of the original data.

Input: A collection of point clouds with their indices and labels (from 'shapes_data.pkl').
Output: Scaled and translated point clouds with original indices and labels, saved in 'scaled_shapes_data.pkl'.
"""
import numpy as np
import pickle

def scale_and_center_point_cloud(point_cloud, target_size=1.0):
    # Calculate the bounding box dimensions
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    bbox_dims = max_coords - min_coords
    
    # Calculate the centroid of the point cloud
    centroid = np.mean(point_cloud, axis=0)
    
    # Translate the centroid of the point cloud to the origin
    translated_point_cloud = point_cloud - centroid

    # Calculate the scale factor (assuming isotropic scaling)
    scale_factor = target_size / np.max(bbox_dims)

    # Scale the point cloud
    scaled_and_centered_point_cloud = translated_point_cloud * scale_factor
    #scaled_and_centered_point_cloud = np.mean(scaled_and_centered_point_cloud, axis=0)
    #print(f"scaled and transleted centroid:{scaled_and_centered_point_cloud}")

    return scaled_and_centered_point_cloud

if __name__ == "__main__":
    # Load the original shape data
    #with open('Data/shapes_data.pkl', 'rb') as file:
    #with open('Shape_data/m_shapes_data.pkl', 'rb') as file:
    with open('Shape_data/m_random_shapes_data.pkl', 'rb') as file:
        original_shape_data = pickle.load(file)

    # Apply the scaling and centering function
    scaled_shape_data = []
    for idx, point_cloud, label in original_shape_data:  # Unpack index, point cloud, and label
        scaled_and_centered_point_cloud = scale_and_center_point_cloud(point_cloud, target_size=2)
        scaled_shape_data.append((idx, scaled_and_centered_point_cloud, label))  # Keep the original index

    # Save the scaled and centered data with indices
    with open('Data/random_scaled_centered_shapes_data.pkl', 'wb') as file:
    #with open('Data/m_scaled_centered_shapes_data1.pkl', 'wb') as file:
        pickle.dump(scaled_shape_data, file)

    print("Scaled and centered shapes and labels saved successfully.")

