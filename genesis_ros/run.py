import genesis as gs
gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.URDF(file='/tmp/genesis_ros/model.urdf', fixed=True)
)

scene.build()

for i in range(1000):
    scene.step()
