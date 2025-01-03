import genesis as gs
from amber.importer.tf import TfImporter, TfImporterConfig
from tf2_amber import (
    TransformStamped,
    Header,
    Transform,
    Time,
    timeFromSec,
    Vector3,
    Quaternion,
)
import math


def get_tf_from_link(cur_t, link):
    TransformStamped(
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


gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(file="/tmp/genesis_ros/model.urdf", fixed=True, pos=(0, 0, 0.4)),
)

scene.build()

print(robot.links)

importer = TfImporter(TfImporterConfig())

for i in range(100):
    importer.write(get_tf_from_link(scene.cur_t, robot.get_link("body_link")))
    importer.write(get_tf_from_link(scene.cur_t, robot.get_link("head_pan_link")))
    scene.step()

importer.finish()
