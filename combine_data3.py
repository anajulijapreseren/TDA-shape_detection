import pickle
import numpy as np

def combine_data(file_path1, file_path2, file_path3=None):
    with open(file_path1, 'rb') as f1, open(file_path2, 'rb') as f2:
        data1 = pickle.load(f1)
        data2 = pickle.load(f2)

    # Check if file_path3 is provided
    if file_path3 is not None:
        with open(file_path3, 'rb') as f3:
            data3 = pickle.load(f3)
        
        # Check if all three files have the same length
        if len(data1) != len(data2) or len(data1) != len(data3):
            raise ValueError("Files do not contain the same number of entries.")
        
        combined_data = []
        for (idx1, vector1, label1), (idx2, vector2, label2), (idx3, vector3, label3) in zip(data1, data2, data3):
            # Check if indices and labels match
            if idx1 != idx2 or label1 != label2 or idx1 != idx3 or label1 != label3:
                raise ValueError(f"Mismatch at index {idx1}: {idx2}, {idx3}, label {label1}: {label2}, {label3}")
            
            # Concatenate vectors
            combined_vector = np.concatenate([vector1, vector2, vector3])
            combined_data.append((idx1, combined_vector, label1))
    else:
        # Check if both file_path1 and file_path2 have the same length
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
file_path1 = 'Data/TR_flattened_images_with_indices.pkl'
file_path2 = 'Data/TR_flattened_images_removed_ball_with_indices.pkl'
file_path3 = 'Data/TR_flattened_images_removed_ball_with_indices_small.pkl'  # Replace with the path to your third file

# Combine the data from three files
combined_data = combine_data(file_path1, file_path2, file_path3)

# Save the combined data
with open('Data/TR_combined3_flattened_data_with_indices.pkl', 'wb') as file:
    pickle.dump(combined_data, file)

print("Combined data saved successfully.")
