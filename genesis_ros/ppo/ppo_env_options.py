from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Union, Optional
from dataclass_wizard import YAMLWizard
from pathlib import Path


@dataclass
class SimulationConfig(YAMLWizard):
    simulate_action_latency: bool = True
    dt: float = 0.02
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 1

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return SimulationConfig().from_yaml_file(path)
        else:
            return SimulationConfig()


@dataclass
class LegPhaseConfig(YAMLWizard):
    period: float = 0.8  # seconds
    offset: float = 0.5  # seconds


@dataclass
class PDCcontrollerConfig(YAMLWizard):
    kp: float = 20.0
    kp_values: Dict[str, float] = field(default_factory=dict)
    kd: float = 0.5
    kd_values: Dict[str, float] = field(default_factory=dict)

    def get_kp(self, joint_name: str) -> float:
        if joint_name in self.kp_values:
            return self.kp_values[joint_name]
        else:
            return self.kp

    def get_kd(self, joint_name: str) -> float:
        if joint_name in self.kd_values:
            return self.kd_values[joint_name]
        else:
            return self.kd


@dataclass
class RewardConfig(YAMLWizard):
    only_positive_rewards: bool = False  # If True, negative rewards are clipped.


@dataclass
class EnvironmentConfig(YAMLWizard):
    default_joint_angles: Dict[str, float] = field(default_factory=dict)
    dof_names: List[str] = field(default_factory=list)
    fix_joints: List[str] = field(default_factory=list)
    foot_links: List[str] = field(default_factory=list)  # Foot links of the robot
    hip_joints: List[str] = field(default_factory=list)  # Hip joints of the robot
    knee_joints: List[str] = field(default_factory=list)  # Knee joints of the robot
    leg_phase: LegPhaseConfig = LegPhaseConfig()
    # PD
    pd_controller: PDCcontrollerConfig = PDCcontrollerConfig()
    # termination
    termination_if_roll_greater_than: float = 10.0  # degree
    termination_if_pitch_greater_than: float = 10.0  # degree
    # base pose
    base_init_pos: Tuple[float, float, float] = field(default=(0.0, 0.0, 0.42))
    base_init_quat: Tuple[float, float, float, float] = field(
        default=(1.0, 0.0, 0.0, 0.0)
    )
    # friction
    friction: Optional[float] = None  # Friction coefficient for all links
    episode_length_seconds: float = 20.0
    resampling_time_seconds: float = 4.0
    action_scale: float = 0.25
    simulate_action_latency: bool = True
    clip_actions: float = 100.0
    reward_config: RewardConfig = RewardConfig()

    def append_joint(self, joint: Tuple[str, float]) -> None:
        if not joint[0] in self.dof_names and joint[0] not in self.fix_joints:
            self.dof_names.append(joint[0])
        if not joint[0] in self.default_joint_angles:
            self.default_joint_angles[joint[0]] = joint[1]

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return EnvironmentConfig().from_yaml_file(path)
        else:
            return EnvironmentConfig()


@dataclass
class ObservationScaleConfig(YAMLWizard):
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return ObservationScaleConfig().from_yaml_file(path)
        else:
            return ObservationScaleConfig()


@dataclass
class ObservationConfig(YAMLWizard):
    obs_scales: ObservationScaleConfig = ObservationScaleConfig()

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return ObservationConfig().from_yaml_file(path)
        else:
            return ObservationConfig()


@dataclass
class CommandConfig(YAMLWizard):
    num_commands: int = 3
    lin_vel_x_range: List[float] = field(default_factory=lambda: [0.5, 0.5])
    lin_vel_y_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    ang_vel_range: List[float] = field(default_factory=lambda: [0.0, 0.0])

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return CommandConfig().from_yaml_file(path)
        else:
            return CommandConfig()
