import genesis as gs
from amber_mcap.importer.tf import TfImporter, TfImporterConfig
from amber_mcap.tf2_amber import (
    TransformStamped,
    Header,
    Transform,
    Time,
    timeFromSec,
    Vector3,
    Quaternion,
)
import math
from genesis_ros.analyze_urdf import get_camera_sensors


def get_tf_from_link(cur_t, link):
    return TransformStamped(
        Header(
            Time(
                int(math.floor(cur_t)),
                int((cur_t - float(math.floor(cur_t))) * pow(10, 9)),
            ),
            "map",
        ),
        link.name,
        Transform(
            Vector3(link.get_pos()[0], link.get_pos()[1], link.get_pos()[2]),
            Quaternion(
                link.get_quat()[0],
                link.get_quat()[1],
                link.get_quat()[2],
                link.get_quat()[3],
            ),
        ),
    )


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True)

    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file="/tmp/genesis_ros/model.urdf", fixed=True, pos=(0, 0, 0.4)),
    )
    print(robot.links)
    for link in robot.links:
        print(link.name)

    camera_sensors = get_camera_sensors("/tmp/genesis_ros/model.urdf", scene, robot)

    scene.build()

    importer = TfImporter(TfImporterConfig())

    for i in range(100):
        importer.write(get_tf_from_link(scene.cur_t, robot.get_link("body_link")))
        importer.write(get_tf_from_link(scene.cur_t, robot.get_link("head_pan_link")))
        scene.step()

    importer.finish()
