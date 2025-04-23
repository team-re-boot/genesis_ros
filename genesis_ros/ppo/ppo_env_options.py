from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SimulationConfig:
    simulate_action_latency: bool = True
    dt: float = 0.02


@dataclass
class EnvironmentConfig:
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
    resampling_time_seconds: float = (
        20.0  # TODO, Enable start training from rosbag data in order to merge real world data.
    )
    action_scale: float = 0.25
    simulate_action_latency: bool = True
    clip_actions: float = 100.0

    def append_joint(self, joint: Tuple[str, float]) -> None:
        if not joint[0] in self.dof_names:
            self.dof_names.append(joint[0])
        if not joint[0] in self.default_joint_angles:
            self.default_joint_angles[joint[0]] = joint[1]

    def append_joints(self, joints: List[Tuple[str, float]]) -> None:
        for joint in joints:
            self.append_joint(joint)


@dataclass
class ObservationScaleConfig:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05


@dataclass
class ObservationConfig:
    obs_scales: ObservationScaleConfig = ObservationScaleConfig()


@dataclass
class RewardConfig:
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3


@dataclass
class CommandConfig:
    num_commands: int = 3
    lin_vel_x_range: List[float] = field(default_factory=lambda: [0.5, 0.5])
    lin_vel_y_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    ang_vel_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
