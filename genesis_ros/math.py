import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union


def get_look_at_point(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Calculates the look-at point from the camera's position and quaternion.

    Args:
        position (np.ndarray): The camera's position vector (x, y, z).
        quaternion (np.ndarray): The camera's rotation quaternion (x, y, z, w).

    Returns:
        np.ndarray: The coordinates of the look-at point vector (x, y, z).
    """
    # Initial forward vector (common in graphics conventions)
    forward_vector: np.ndarray = np.array([0, 0, -1])

    # Convert the quaternion to a scipy Rotation object
    rotation: R = R.from_quat(quaternion)

    # Rotate the forward vector
    rotated_forward_vector: np.ndarray = rotation.apply(forward_vector)

    # Calculate the look-at point
    look_at_point: np.ndarray = position + rotated_forward_vector

    return look_at_point
