import genesis as gs  # type: ignore
from genesis_ros.genesis_ros_env_options import (
    SimulationConfig,
    EnvironmentConfig,
    ObservationConfig,
    RewardConfig,
    CommandConfig,
)
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from typing import Any
import functools
import inspect
import sys
import torch


def genesis_entity(func) -> Any:
    """
    Decorator to check if the return type is gs.morphs.Morph
    """
    func._is_genesis_entity = True

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, gs.morphs.Morph):
            raise TypeError(
                f"The return type of function {func.__name__} is not gs.morphs.Morph."
            )
        return result

    return wrapper


def list_genesis_entities(module=sys.modules[__name__]):
    decorated_functions = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if hasattr(func, "_is_genesis_entity"):
            decorated_functions.append(name)
    return decorated_functions


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
        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="/tmp/genesis_ros/model.urdf",
                fixed=True,
                pos=self.env_cfg.base_init_pos,
                merge_fixed_links=False,
            ),
        )
        self.base_init_pos = torch.tensor(self.env_cfg.base_init_pos, device=gs.device)
        self.base_init_quat = torch.tensor(
            self.env_cfg.base_init_quat, device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        for function_name in list_genesis_entities():
            function = getattr(sys.modules[__name__], function_name)
            self.scene.add_entity(function())

        # build
        self.scene.build(n_envs=num_envs)
        # names to indices
        for joint in self.robot.joints:
            if self.robot.get_joint(joint.name).dof_idx_local:
                self.env_cfg.append_joint(
                    (
                        joint.name,
                        self.robot.get_dofs_position(
                            [self.robot.get_joint(joint.name).dof_idx_local]
                        ).item(),
                    )
                )
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local for name in self.env_cfg.dof_names
        ]


if __name__ == "__main__":
    gs.init(logging_level="warning", backend=gs.cpu)

    @genesis_entity
    def add_plane():
        return gs.morphs.Plane()

    env = GenesisRosEnv(
        1,
        SimulationConfig(),
        EnvironmentConfig(),
        ObservationConfig(),
        RewardConfig(),
        CommandConfig(),
        "/tmp/genesis_ros/model.urdf",
    )
