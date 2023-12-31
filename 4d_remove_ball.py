import numpy as np
import pickle

def remove_center_ball_4d(point_cloud, radius=0.5):
    """
    Removes points within a given radius from the center of the 4D point cloud.

    :param point_cloud: Numpy array representing the 4D point cloud.
    :param radius: Radius of the 4D ball to be removed.
    :return: Numpy array of the 4D point cloud with the central 4D ball removed.
    """
    distance_from_center = np.sqrt(np.sum(point_cloud**2, axis=1))
    return point_cloud[distance_from_center > radius]

if __name__ == "__main__":
#-----------------REMOVE BIG BALL-------------------------------------------
    # Load the scaled shape data for 4D
    with open('Data/4d_scaled_centered_shapes_data.pkl', 'rb') as file:
        scaled_shape_data_4d = pickle.load(file)

    # Process each 4D shape to remove the center 4D ball
    modified_shape_data_4d = []
    for idx, point_cloud, label in scaled_shape_data_4d:
        modified_point_cloud = remove_center_ball_4d(point_cloud, radius=0.5)
        modified_shape_data_4d.append((idx, modified_point_cloud, label))

    # Save the modified 4D data
    with open('Data/4d_modified_shapes_data.pkl', 'wb') as file:
        pickle.dump(modified_shape_data_4d, file)

    print("Modified 4D shapes (with BIG center 4D ball removed) and labels saved successfully.")



    #-----------------REMOVE SMALL BALL-------------------------------------------
    # Load the scaled shape data
    with open('Data/4d_scaled_centered_shapes_data.pkl', 'rb') as file:
    #with open('Data/m_scaled_centered_shapes_data1.pkl', 'rb') as file:
        scaled_shape_data_4d = pickle.load(file)

    # Process each shape to remove the center ball
    modified_shape_data_4d = []
    for idx, point_cloud, label in scaled_shape_data_4d:
        modified_point_cloud = remove_center_ball_4d(point_cloud,radius=0.2)
        modified_shape_data_4d.append((idx, modified_point_cloud, label))

    # Save the modified data
    with open('Data/4d_modified_shapes_data_small.pkl', 'wb') as file:
    #with open('Data/m_modified_shapes_data1.pkl', 'wb') as file:
        pickle.dump(modified_shape_data_4d, file)

    print("Modified 4D shapes (with SMALL center 4D ball removed) and labels saved successfully.")