#This file opens a file with already generated shapes, calculates their persistence diagrams, their persistence images (end flattens them) and then 
#saves flattened PIs in 'Shape_detection/flattened_images.pkl' and thir labels in 'Shape_detection/labels.pkl'
#they are saved in the same order

import pickle
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage
import numpy as np

# Load the shapes and labels
with open('shapes_data.pkl', 'rb') as file:
    shape_data = pickle.load(file)

def create_flattened_PI(shape_data):
    # Initialize VietorisRipsPersistence
    homology_dimensions = [0, 1, 2]
    persistence = VietorisRipsPersistence(metric="euclidean",
                                        homology_dimensions=homology_dimensions,
                                        n_jobs=12,
                                        collapse_edges=True)

    # Initialize the PersistenceImage transformer
    persistence_image = PersistenceImage(n_bins=50, weight_function=lambda x: x[1])

    # Process each shape
    flattened_images = []
    labels = []
    for i, (point_cloud, label) in enumerate(shape_data):
        # Reshape for Giotto-tda
        X = point_cloud.reshape(1, *point_cloud.shape)

        # Compute the persistence diagrams
        diagrams = persistence.fit_transform(X)

        # Transform the persistence diagrams to persistence images
        images = persistence_image.fit_transform(diagrams)

        # Flatten the images
        flattened_image = images.reshape(images.shape[0], -1)

        # Collect the results
        flattened_images.append(flattened_image[0])  # [0] to remove the extra dimension
        labels.append(label)

        # Print progress every 5 shapes
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1} shapes.")

    # Convert to numpy arrays
    flattened_images = np.array(flattened_images)
    labels = np.array(labels)

    return flattened_images,labels

def create_flattened_PI_no_labels(shape_data):
    """
    Process a list of shapes to compute their persistence diagrams and images,
    and then flatten these images for SVM.

    Args:
    shape_data (list of tuples): Each tuple contains (point cloud, label).

    Returns:
    Tuple of (flattened_images, labels)
    """
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage
    import numpy as np

    # Initialize VietorisRipsPersistence
    homology_dimensions = [0, 1, 2]
    persistence = VietorisRipsPersistence(metric="euclidean",
                                          homology_dimensions=homology_dimensions,
                                          n_jobs=12,
                                          collapse_edges=True)

    # Initialize the PersistenceImage transformer
    persistence_image = PersistenceImage(n_bins=50, weight_function=lambda x: x[1])

    # Process each shape
    flattened_images = []
    for i, (point_cloud) in enumerate(shape_data):
        # Reshape for Giotto-tda
        X = point_cloud.reshape(1, *point_cloud.shape)

        # Compute the persistence diagrams
        diagrams = persistence.fit_transform(X)

        # Transform the persistence diagrams to persistence images
        images = persistence_image.fit_transform(diagrams)

        # Flatten the images
        flattened_image = images.reshape(images.shape[0], -1)

        # Collect the results
        flattened_images.append(flattened_image[0])  # [0] to remove the extra dimension

        # Print progress every 5 shapes
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1} shapes.")

    # Convert to numpy arrays
    flattened_images = np.array(flattened_images)

    return flattened_images

if __name__ == "__main__":

    flattened_images,labels=create_flattened_PI(shape_data=shape_data)

    # Save the flattened images and labels
    with open('flattened_images.pkl', 'wb') as f_img, open('labels.pkl', 'wb') as f_lbl:
        pickle.dump(flattened_images, f_img)
        pickle.dump(labels, f_lbl)

    print("All shapes processed and results saved.")
