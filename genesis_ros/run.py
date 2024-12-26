import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(file="/tmp/genesis_ros/model.urdf", fixed=True, pos=(0, 0, 0.4)),
)

scene.build()

print(robot.links)
body_link = robot.get_link("body_link")
head_pan_link = robot.get_link("head_pan_link")

for i in range(1000):
    # print(robot.get_links_pos())
    print(body_link.get_pos())
    print(head_pan_link.get_pos())
    scene.step()
