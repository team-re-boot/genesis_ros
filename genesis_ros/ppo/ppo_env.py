import genesis as gs  # type: ignore
from genesis_ros.ppo.ppo_env_options import (
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
from typing import Any, List, Tuple
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


def list_genesis_entities(module=sys.modules[__name__]) -> List[str]:
    decorated_functions = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if hasattr(func, "_is_genesis_entity"):
            decorated_functions.append(name)
    return decorated_functions


def ppo_reward_function(func) -> Any:
    """
    Decorator for reward functions
    """
    func._is_ppo_reward_function = True

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                f"The return type of function {func.__name__}  is not torch.Tensor."
            )
        return result

    return wrapper


def list_ppo_reward_functions(module=sys.modules[__name__]) -> List[str]:
    decorated_functions = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if hasattr(func, "_is_ppo_reward_function"):
            decorated_functions.append(name)
    return decorated_functions


class PPOEnv:
    def __init__(
        self,
        num_envs: int,
        simulation_cfg: SimulationConfig,
        env_cfg: EnvironmentConfig,
        obs_cfg: ObservationConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        urdf_path: str = "/tmp/genesis_ros/model.urdf",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
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

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
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
                file=urdf_path,
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
        self.motor_dofs = [0] + [
            self.robot.get_joint(name).dof_idx_local for name in self.env_cfg.dof_names
        ]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg.kp] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg.kd] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions = {}  # type: ignore
        self.episode_sums = {}  # type: ignore
        for ppo_reward_function in list_ppo_reward_functions():
            print("Adding reward function: ", ppo_reward_function)
            setattr(
                self,
                "_" + ppo_reward_function,
                getattr(sys.modules[__name__], "ppo_reward_function"),
            )
            self.reward_cfg.reward_scales[ppo_reward_function] *= self.dt
            self.reward_functions[ppo_reward_function] = getattr(
                self, "_" + ppo_reward_function
            )
            self.episode_sums[ppo_reward_function] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # initialize buffers
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg.default_joint_angles[name]
                for name in self.env_cfg.dof_names
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        # extra information for logging
        self.extras = dict()  # type: ignore
        self.extras["observations"] = dict()


if __name__ == "__main__":
    gs.init(logging_level="warning", backend=gs.cpu)

    @genesis_entity
    def add_plane():
        return gs.morphs.Plane()

    # ------------ reward functions----------------
    @ppo_reward_function
    def reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg.tracking_sigma)

    @ppo_reward_function
    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg.tracking_sigma)

    @ppo_reward_function
    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    @ppo_reward_function
    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    @ppo_reward_function
    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    @ppo_reward_function
    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg.base_height_target)

    reward_config = RewardConfig()
    reward_config.reward_scales = {
        "reward_tracking_lin_vel": 1.0,
        "reward_tracking_ang_vel": 0.2,
        "reward_lin_vel_z": -1.0,
        "reward_action_rate": -0.005,
        "reward_similar_to_default": -0.1,
        "reward_base_height": -50.0,
    }

    env = PPOEnv(
        1,
        SimulationConfig(),
        EnvironmentConfig(),
        ObservationConfig(),
        reward_config,
        CommandConfig(),
        "urdf/go2/urdf/go2.urdf",
    )
