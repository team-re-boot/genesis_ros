from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import genesis as gs


@dataclass
class SimulationConfig:
    simulate_action_latency: bool = True
    dt: float = 0.02


@dataclass
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
    num_obs: int = 45
    obs_scales: ObservationScaleConfig = ObservationScaleConfig()


# TODO: Support decorator for describe reward function and remove this config
@dataclass
class RewardScalesConfig:
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.2
    lin_vel_z: float = -1.0
    base_height: float = -50.0
    action_rate: float = -0.005
    similar_to_default: float = -0.1


@dataclass
class RewardConfig:
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3
    reward_scales: RewardScalesConfig = RewardScalesConfig()


@dataclass
class CommandConfig:
    num_commands: int = 3
    lin_vel_x_range: List[float] = field(default_factory=lambda: [0.5, 0.5])
    lin_vel_y_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    ang_vel_range: List[float] = field(default_factory=lambda: [0.0, 0.0])


class GenesisRosEnv:
    def __init__(
        self,
        num_envs: int,
        simulation_cfg: SimulationConfig,
        env_cfg: EnvironmentConfig,
        obs_cfg: ObservationConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        urdf_path: str = "/tmp/genesis_ros/model.urdf",
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg.num_obs
        self.num_privileged_obs = None
        self.num_actions = env_cfg.num_actions
        self.num_commands = command_cfg.num_commands
        self.simulate_action_latency = (
            simulation_cfg.simulate_action_latency
        )  # there is a 1 step latency on real robot
        self.dt = simulation_cfg.dt

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg.obs_scales
        self.reward_scales = reward_cfg.reward_scales

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
