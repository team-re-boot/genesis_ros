import genesis as gs
from genesis_ros.rosbag_writer import RosbagWriter
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


def get_header(cur_t, frame_id):
    return Header(
        Time(
            int(math.floor(cur_t)),
            int((cur_t - float(math.floor(cur_t))) * pow(10, 9)),
        ),
        frame_id,
    )


def get_tf_from_link(cur_t, link):
    return TransformStamped(
        get_header(cur_t, "map"),
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
        gs.morphs.URDF(
            file="/tmp/genesis_ros/model.urdf",
            fixed=True,
            pos=(0, 0, 0.4),
            merge_fixed_links=False,
        ),
    )

    # camera_sensors = get_camera_sensors("/tmp/genesis_ros/model.urdf", scene, robot)

    scene.build()

    rosbag_writer = RosbagWriter()

    for i in range(100):
        rosbag_writer.write_tf(
            get_tf_from_link(scene.cur_t, robot.get_link("body_link"))
        )
        rosbag_writer.write_tf(
            get_tf_from_link(scene.cur_t, robot.get_link("head_pan_link"))
        )
        # for camera in camera_sensors:
        #     rosbag_writer.write_image(
        #         "image_raw",
        #         camera.update()[0],
        #         get_header(scene.cur_t, "cam_gazebo_link"),
        #     )
        scene.step()

    rosbag_writer.finish()
