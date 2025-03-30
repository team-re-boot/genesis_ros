import xml.etree.ElementTree as ET
from pathlib import Path
from amber_mcap.tf2_amber import TransformStamped
from genesis_ros import math


class CameraSensor:
    def __init__(self, link: str):
        self.link = link

    def get_look_at_point(gs_link):
        return math.get_look_at_point(
            np.array(link.get_pos()), np.array(link.get_quat())
        )


def get_camera_sensors(urdf_path: Path):
    tree = ET.parse(urdf_path)
    gazebo_elements = tree.getroot().findall(".//gazebo")
    if gazebo_elements:
        for gazebo_element in gazebo_elements:
            sensor_elements = gazebo_element.findall(".//sensor")
            if sensor_elements:
                for sensor_element in sensor_elements:
                    if sensor_element.attrib["type"] == "camera":
                        CameraSensor(gazebo_element.attrib["reference"])
