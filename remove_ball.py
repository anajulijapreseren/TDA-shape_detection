"""
This script includes a function 'remove_center_ball' designed to process 3D point clouds by removing a specified spherical region 
from the origin. 

The main execution block of the script, when run, applies this function to a collection of pre-processed point clouds that are 
already scaled and centered around the origin (0, 0, 0). These point clouds are loaded from 'scaled_centered_shapes_data.pkl', 
which includes the original indices and labels.

Each modified point cloud, with the central ball of points removed, is then appended to a list along with its original index and label. 
This modified data is subsequently saved in a new file named 'modified_shapes_data.pkl'. This file maintains the order and classification 
of the original data while introducing the specified modification.

Input: Scaled and centered point clouds with original indices and labels (from 'scaled_centered_shapes_data.pkl').
Output: Point clouds with a central spherical region removed, along with original indices and labels, saved in 'modified_shapes_data.pkl'.
"""



import numpy as np
import pickle

def remove_center_ball(point_cloud, radius=0.5):
    """
    Removes points within a given radius from the center of the point cloud.

    :param point_cloud: Numpy array representing the point cloud.
    :param radius: Radius of the ball to be removed.
    :return: Numpy array of the point cloud with the central ball removed.
    """
    distance_from_center = np.sqrt(np.sum(point_cloud**2, axis=1))
    return point_cloud[distance_from_center > radius]

if __name__ == "__main__":
    # Load the scaled shape data
    #with open('Data/scaled_centered_shapes_data.pkl', 'rb') as file:
    with open('Data/m_scaled_centered_shapes_data1.pkl', 'rb') as file:
        scaled_shape_data = pickle.load(file)

    # Process each shape to remove the center ball
    modified_shape_data = []
    for idx, point_cloud, label in scaled_shape_data:
        modified_point_cloud = remove_center_ball(point_cloud)
        modified_shape_data.append((idx, modified_point_cloud, label))

    # Save the modified data
    #with open('Data/modified_shapes_data.pkl', 'wb') as file:
    with open('Data/m_modified_shapes_data1.pkl', 'wb') as file:
        pickle.dump(modified_shape_data, file)

    print("Modified shapes (with center ball removed) and labels saved successfully.")

