import xml.etree.ElementTree as ET
from pathlib import Path


class CameraSensor:
    def __init__(self, link: str):
        self.link = link


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
