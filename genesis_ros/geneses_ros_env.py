# class Simulation:
#     def __init__(self, ):
#         gs.init(backend=gs.cpu)
#         self.__hash__scene = gs.Scene(show_viewer=True)
#         self.plane = scene.add_entity(gs.morphs.Plane())
import genesis as gs
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclasses
class EnvironmentConfig:
    num_actions: int = 12
    default_joint_angles: Dict[str, float] = field(default_factory=dict)
    dof_names: List[str] = field(default_factory=list)
    # PD
    kp: float = 20.0
    kd: float = 0.5
    # termination
    termination_if_roll_greater_than: float = 10  # degree
    termination_if_pitch_greater_than: float = 10.0  # degree
    # base pose
    base_init_pos: Tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    base_init_quat: Tuple[float, float, float, float] = field(
        default=(1.0, 0.0, 0.0, 0.0)
    )
    episode_length_seconds: float = 20.0
    resampling_time_seconds: float = 20.0  # TODO, Enable start training from rosbag data in order to merge real world data.
    action_scale: float = 0.25
    simulate_action_latency: bool = True
    clip_actions: float = 100.0

    def append_joint(self, joint: Tuple(str, float)) -> None:
        if not joint[0] in self.dof_names:
            self.dof_names.append(joint[0])
        if not joint[0] in self.default_joint_angles:
            self.default_joint_angles[joint[0]] = joint[1]

    def append_joints(self, joints: List[Tuple(str, float)]) -> None:
        for joint in joints:
            self.append_joint(joint)


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
