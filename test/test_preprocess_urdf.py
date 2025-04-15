from genesis_ros.preprocess_urdf import find_ros2_packages, save_urdf_to_tmp
import pytest
import os


@pytest.mark.skipif(
    os.environ.get("ROS_DISTRO") is None, reason="ROS 2 was not installed"
)
def test_find_ros2_packages():
    find_ros2_packages()


@pytest.mark.skipif(
    os.environ.get("ROS_DISTRO") is None
    or "op3_description" not in find_ros2_packages(),
    reason="ROS 2 was not installed or op3_description package was not installed.",
)
def test_save_urdf_to_tmp():
    urdf_path = find_ros2_packages()["op3_description"] + "/urdf/robotis_op3.urdf"
    try:
        with open(urdf_path, "r") as urdf_file:
            urdf_content = urdf_file.read()
    except FileNotFoundError:
        print(f"The file {urdf_path} was not found.")
        return
    except IOError as e:
        print(f"Failed to read the URDF file: {e}")
        return
    save_urdf_to_tmp(urdf_content)
