import xml.etree.ElementTree as ET
from pathlib import Path
from amber_mcap.tf2_amber import TransformStamped
from genesis_ros import math
import genesis as gs


class CameraSensor:
    def __init__(self, link_name: str, gs_scene, gs_robot):
        self.link_name = link_name
        self.gs_scene = gs_scene
        self.gs_robot = gs_robot
        self.gs_scene.add_camera(
            res=(640, 480),
            pos=self.get_camera_position(),
            lookat=self.get_look_at_point(),
            fov=30,
        )

    def get_camera_position(self):
        return self.gs_robot.get_link(self.link_name).get_pos()

    def get_look_at_point(self):
        gs_link = self.gs_robot.get_link(self.link_name)
        return math.get_look_at_point(
            np.array(gs_link.get_pos()), np.array(gs_link.get_quat())
        )


def get_camera_sensors(urdf_path: Path, gs_scene, gs_robot):
    tree = ET.parse(urdf_path)
    gazebo_elements = tree.getroot().findall(".//gazebo")
    if gazebo_elements:
        for gazebo_element in gazebo_elements:
            sensor_elements = gazebo_element.findall(".//sensor")
            if sensor_elements:
                for sensor_element in sensor_elements:
                    if sensor_element.attrib["type"] == "camera":
                        CameraSensor(
                            gazebo_element.attrib["reference"], gs_scene, gs_robot
                        )
