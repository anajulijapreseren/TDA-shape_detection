import numpy as np
import pickle
import time
from multiprocessing import Pool
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage

def process_shape_4d(data):
    """
    Process a single 4D shape to compute its persistence diagram and image,
    and then flatten the image.

    Args:
    data (tuple): Contains (index, 4D point cloud, label).

    Returns:
    Tuple of (index, flattened_image, label)
    """
    idx, point_cloud, label = data
    # Initialize VietorisRipsPersistence for 4D
    persistence = VietorisRipsPersistence(metric="euclidean",
                                          homology_dimensions=[0, 1, 2, 3],  # Adjust dimensions for 4D
                                          n_jobs=1,
                                          collapse_edges=True)

    # Initialize the PersistenceImage transformer
    persistence_image = PersistenceImage(n_bins=50, weight_function=lambda x: x[1])

    # Reshape for Giotto-tda
    X = point_cloud.reshape(1, *point_cloud.shape)

    # Compute the persistence diagrams
    diagrams = persistence.fit_transform(X)

    # Transform the persistence diagrams to persistence images
    images = persistence_image.fit_transform(diagrams)

    # Flatten the images
    flattened_image = images.reshape(images.shape[0], -1)[0]

    return (idx, flattened_image, label)

def parallel_process_shapes_4d(shape_data, num_processes=8):
    """
    Process 4D shapes in parallel to compute persistence diagrams and images.

    Args:
    shape_data (list of tuples): Each tuple contains (index, 4D point cloud, label).
    num_processes (int): Number of parallel processes to use.

    Returns:
    List of tuples containing (index, flattened_image, label).
    """
    with Pool(num_processes) as pool:
        results = pool.map(process_shape_4d, shape_data)
    return results

if __name__ == "__main__":
    start_time = time.time()
    #----------------------------------------ORIGINAL----------------------------------------------------------------
    # # Load the scaled and centered shape data
    # with open('Data/4d_scaled_centered_shapes_data.pkl', 'rb') as file:
    # #with open('Data/m_scaled_centered_shapes_data1.pkl', 'rb') as file:
    #     shape_data = pickle.load(file)

    # # Start timer
    # start_time = time.time()

    # # Process shapes in parallel
    # processed_data = parallel_process_shapes_4d(shape_data)

    # # Save the results including indices
    # with open('Data/4d_flattened_images_with_indices.pkl', 'wb') as f:
    # #with open('Data/m_flattened_images_with_indices1.pkl', 'wb') as f:
    #     pickle.dump(processed_data, f)

    # print("All original 4D shapes processed and results saved.")
    
    #----------------------------------------REMOVED BIG BALL----------------------------------------------------------------
    # Load the scaled and centered shape data without the ball of points
    with open('Data/4d_modified_shapes_data.pkl', 'rb') as file:
    #with open('Data/m_modified_shapes_data1.pkl', 'rb') as file:
        shape_data = pickle.load(file)

    # Process shapes in parallel
    processed_data = parallel_process_shapes_4d(shape_data)

    # Save the results including indices
    with open('Data/4d_flattened_images_removed_ball_with_indices.pkl', 'wb') as f:
    #with open('Data/m_flattened_images_removed_ball_with_indices1.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

    print("All shapes without BIG ball of points processed and results saved.")

    #----------------------------------------REMOVED SMALL BALL----------------------------------------------------------------
    # Load the scaled and centered shape data without the ball of points
    with open('Data/4d_modified_shapes_data_small.pkl', 'rb') as file:
    #with open('Data/m_modified_shapes_data1.pkl', 'rb') as file:
        shape_data = pickle.load(file)

    # Process shapes in parallel
    processed_data = parallel_process_shapes_4d(shape_data)

    # Save the results including indices
    with open('Data/4d_flattened_images_removed_ball_with_indices_small.pkl', 'wb') as f:
    #with open('Data/m_flattened_images_removed_ball_with_indices1.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

    print("All shapes without SMALL ball of points processed and results saved.")

    # End timer
    end_time = time.time()

    # Calculate duration
    duration = end_time - start_time
    print(f"Processing time: {duration:.2f} seconds")

