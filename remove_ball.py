import numpy as np
import pickle

def remove_center_ball(point_cloud, radius=0.3):
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
    with open('scaled_centered_shapes_data.pkl', 'rb') as file:
        scaled_shape_data = pickle.load(file)

    # Process each shape to remove the center ball
    modified_shape_data = []
    for idx, point_cloud, label in scaled_shape_data:
        modified_point_cloud = remove_center_ball(point_cloud)
        modified_shape_data.append((idx, modified_point_cloud, label))

    # Save the modified data
    with open('modified_shapes_data.pkl', 'wb') as file:
        pickle.dump(modified_shape_data, file)

    print("Modified shapes (with center ball removed) and labels saved successfully.")

