import pickle
import numpy as np

def combine_data(file_path1, file_path2):
    with open(file_path1, 'rb') as f1, open(file_path2, 'rb') as f2:
        data1 = pickle.load(f1)
        data2 = pickle.load(f2)

    # Check if both files have the same length
    if len(data1) != len(data2):
        raise ValueError("Files do not contain the same number of entries.")

    combined_data = []
    for (idx1, vector1, label1), (idx2, vector2, label2) in zip(data1, data2):
        # Check if indices and labels match
        if idx1 != idx2 or label1 != label2:
            raise ValueError(f"Mismatch at index {idx1}: {idx2}, label {label1}: {label2}")
        # Concatenate vectors
        combined_vector = np.concatenate([vector1, vector2])
        combined_data.append((idx1, combined_vector, label1))

    return combined_data

# Paths to your files
# file_path1 = 'Data/flattened_images_with_indices.pkl'
# file_path2 = 'Data/flattened_images_removed_ball_with_indices.pkl'
file_path1 = 'Data/m_flattened_images_with_indices.pkl'
file_path2 = 'Data/m_flattened_images_removed_ball_with_indices.pkl'

# Combine the data
combined_data = combine_data(file_path1, file_path2)

# Save the combined data
#with open('Data/combined_flattened_data_with_indices.pkl', 'wb') as file:
with open('Data/m_combined_flattened_data_with_indices.pkl', 'wb') as file:
    pickle.dump(combined_data, file)

print("Combined data saved successfully.")
