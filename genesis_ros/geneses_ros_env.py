# class Simulation:
#     def __init__(self, ):
#         gs.init(backend=gs.cpu)
#         self.__hash__scene = gs.Scene(show_viewer=True)
#         self.plane = scene.add_entity(gs.morphs.Plane())
import genesis as gs


class GenesisRosEnv:
    def __init__(
        self,
        num_envs: int,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        urdf_path: str = "/tmp/genesis_ros/model.urdf",
        show_viewer=False,
        device="cuda",
    ):
        pass
