import xml.etree.ElementTree as ET
from pathlib import Path
from amber_mcap.tf2_amber import TransformStamped
from genesis_ros import math
import genesis as gs
import numpy as np
from typing import List


class CameraSensor:
    def __init__(self, link_name: str, gs_scene, gs_robot):
        self.link_name = link_name
        self.gs_robot = gs_robot
        self.gs_camera = gs_scene.add_camera(
            res=(640, 480),
            pos=(0, 0, 0),
            lookat=(0, 0, 1),
            fov=30,
        )

    def update(self):
        self.gs_camera.set_pose(
            pos=self.get_camera_position(), lookat=self.get_look_at_point()
        )
        return self.gs_camera.render()

    def get_camera_position(self):
        return self.gs_robot.get_link(self.link_name).get_pos()

    def get_look_at_point(self):
        gs_link = self.gs_robot.get_link(self.link_name)
        return math.get_look_at_point(
            np.array(gs_link.get_pos()), np.array(gs_link.get_quat())
        )


def get_camera_sensors(urdf_path: Path, gs_scene, gs_robot) -> List[CameraSensor]:
    camera_sensors = []
    tree = ET.parse(urdf_path)
    gazebo_elements = tree.getroot().findall(".//gazebo")
    if gazebo_elements:
        for gazebo_element in gazebo_elements:
            sensor_elements = gazebo_element.findall(".//sensor")
            if sensor_elements:
                for sensor_element in sensor_elements:
                    if sensor_element.attrib["type"] == "camera":
                        camera_sensors.append(
                            CameraSensor(
                                gazebo_element.attrib["reference"], gs_scene, gs_robot
                            )
                        )
    return camera_sensors
