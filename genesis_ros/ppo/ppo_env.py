import genesis as gs  # type: ignore
from genesis_ros.ppo.ppo_env_options import (
    SimulationConfig,
    EnvironmentConfig,
    ObservationConfig,
    CommandConfig,
)
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from typing import Any, List, Tuple, Optional, Dict, Callable
import functools
import inspect
import sys
import torch
import math
import types


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class PPOEnv:
    def __init__(
        self,
        entities: List[gs.morphs.Morph],
        reward_functions: List[Tuple[Callable[[Any], torch.tensor], float]],
        num_envs: int,
        simulation_cfg: SimulationConfig,
        env_cfg: EnvironmentConfig,
        obs_cfg: ObservationConfig,
        command_cfg: CommandConfig,
        urdf_path: str = "/tmp/genesis_ros/model.urdf",
        show_viewer: bool = False,
    ):
        self.device = gs.device
        self.num_envs = num_envs
        self.num_privileged_obs = None
        self.num_commands = command_cfg.num_commands
        self.simulate_action_latency = (
            simulation_cfg.simulate_action_latency
        )  # there is a 1 step latency on real robot
        self.dt = simulation_cfg.dt
        self.max_episode_length = math.ceil(env_cfg.episode_length_seconds / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
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
            show_viewer=show_viewer,
        )
        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                fixed=False,
                pos=self.env_cfg.base_init_pos,
                quat=self.env_cfg.base_init_quat,
                merge_fixed_links=False,
            ),
        )
        self.base_init_pos = torch.tensor(
            self.env_cfg.base_init_pos, device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg.base_init_quat, device=self.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        for entity in entities:
            if isinstance(entity, gs.morphs.Morph):
                self.scene.add_entity(entity)
            else:
                raise TypeError(
                    f"Entity {entity} is not a valid Genesis morph. Please check the entity type."
                )

        # build
        self.scene.build(n_envs=num_envs)
        # names to indices
        for joint in self.robot.joints:
            if self.robot.get_joint(joint.name).dof_idx_local:
                if self.robot.base_joint.name == joint.name:
                    continue
                self.env_cfg.append_joint(
                    (
                        joint.name,
                        self.robot.get_dofs_position(
                            [self.robot.get_joint(joint.name).dof_idx_local]
                        )[0].item(),
                    )
                )
        self.motors_dof_idx = [
            self.robot.get_joint(name).dof_start for name in self.env_cfg.dof_names
        ]
        print("Number of joints: ", len(self.env_cfg.dof_names))
        print("Joints : ", self.env_cfg.dof_names)
        self.num_actions = len(self.env_cfg.dof_names)
        print("Number of actions: ", self.num_actions)
        self.num_obs = 9 + 3 * self.num_actions
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local for name in self.env_cfg.dof_names
        ]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg.kp] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg.kd] * self.num_actions, self.motor_dofs)
        # Set default joint angle as robot position
        default_joint_angles = [
            self.env_cfg.default_joint_angles[name] for name in self.env_cfg.dof_names
        ]

        self.robot.set_dofs_position(
            [default_joint_angles[:] for _ in range(num_envs)],
            self.motor_dofs,
            zero_velocity=True,
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions = {}  # type: ignore
        self.reward_scales: Dict[str, float] = {}
        self.episode_sums = {}  # type: ignore
        for reward_function, reward_scale in reward_functions:
            print("Adding reward function: ", reward_function.__name__)
            print("Reward_scale = ", reward_scale)
            print("Reward scale considering time delta = ", reward_scale * self.dt)
            setattr(
                self,
                "_" + reward_function.__name__,
                types.MethodType(reward_function, self),
            )
            self.reward_scales[reward_function.__name__] = reward_scale * self.dt
            self.reward_functions[reward_function.__name__] = getattr(
                self, "_" + reward_function.__name__
            )
            self.episode_sums[reward_function.__name__] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )
        print("Reward functions setup finished.")

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

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg.lin_vel_x_range, (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg.lin_vel_y_range, (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg.ang_vel_range, (len(envs_idx),), self.device
        )

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg.clip_actions, self.env_cfg.clip_actions
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = exec_actions * self.env_cfg.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()
        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (
                self.episode_length_buf
                % int(self.env_cfg.resampling_time_seconds / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg.termination_if_pitch_greater_than
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg.termination_if_roll_greater_than
        )
        # print("Pitch", self.base_euler[:, 1])
        # print("Roll", self.base_euler[:, 0])

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales.dof_pos,  # self.num_actions
                self.dof_vel * self.obs_scales.dof_vel,  # self.num_actions
                self.actions,  # self.num_actions
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg.episode_length_seconds
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None
