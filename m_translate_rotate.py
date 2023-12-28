import numpy as np
from scipy.spatial.transform import Rotation


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


