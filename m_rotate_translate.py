import numpy as np
from scipy.spatial.transform import Rotation
import random
import pickle

def translate(point_cloud, translation_vector):
    """
    point_cloud: numpy array
    translation vector: numpy array
    """
    translated = point_cloud + translation_vector
    return translated


def rotate(point_cloud, rotation_axis, phi):
    """
    point_cloud: numpy array
    rotation_axis: numpy array
    phi (rad)
    """
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation = Rotation.from_rotvec(phi * rotation_axis)
    rotated = rotation.apply(point_cloud)
    return rotated





with open('Shape_data/m_shapes_data.pkl', 'rb') as file:
        original_shape_data = pickle.load(file)


rotated_shapes = []
rotated_data = {}
for index, array, label in original_shape_data:
    # random translation:
    t_vec = np.random.uniform(-20,20,3)
    # random rotation
    r_vec = np.random.uniform(-1,1,3)
    phi = random.random() * 2 * np.pi
    rotated = rotate(array, r_vec, phi)
    translated = translate(rotated, t_vec)
    rotated_shapes.append((index, translated, label))
    rotated_data[index] = f"rotation: {r_vec}, {phi}, translation: {t_vec}"
    
with open('Shape_data/m_rotated_translated_shapes_data.pkl', 'wb') as file:
    pickle.dump(rotated_shapes, file)

with open('Shape_data/m_rotated_translated_rotation_data.txt', 'w') as file:
        for label, rt in rotated_data.items():
            file.write(f"{label}:{rt}\n")
    
