from genesis_ros.ppo.ppo_env import PPOEnv
import genesis as gs  # type: ignore
import torch
from genesis_ros.ppo.ppo_env_options import (
    SimulationConfig,
    EnvironmentConfig,
    ObservationConfig,
    RewardConfig,
    CommandConfig,
)
from genesis_ros.ppo.ppo_train_options import TrainConfig, Algorithm, Policy, Runner
import pickle
import shutil
import os
from rsl_rl.runners import OnPolicyRunner
from dataclasses import asdict
from pathlib import Path
import argparse


def train(
    device: str = "gpu",
    config_directory: Path = Path(__file__).resolve(),
    num_environments: int = 4096,
    urdf_path: str = "urdf/go2/urdf/go2.urdf",
):
    if device == "cpu":
        gs.init(logging_level="warning", backend=gs.cpu)
    elif device == "gpu":
        gs.init(logging_level="warning", backend=gs.gpu)
    else:
        raise ValueError("Invalid device specified. Choose 'cpu' or 'gpu'.")

    reward_functions = []

    # ------------ reward functions----------------
    def reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg.tracking_sigma)

    reward_functions.append((reward_tracking_lin_vel, 1.0))

    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg.tracking_sigma)

    reward_functions.append((reward_tracking_ang_vel, 0.2))

    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    reward_functions.append((reward_lin_vel_z, -1.0))

    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    reward_functions.append((reward_action_rate, -0.005))

    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    reward_functions.append((reward_similar_to_default, -0.1))

    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg.base_height_target)

    reward_functions.append((reward_base_height, -50.0))

    env_cfg = EnvironmentConfig()
    if (config_directory / "environment_config.yaml").exists():
        env_cfg = EnvironmentConfig().from_yaml_file(
            config_directory / "environment_config.yaml"
        )
    sim_cfg = SimulationConfig()
    obs_cfg = ObservationConfig()
    reward_cfg = RewardConfig()
    command_cfg = CommandConfig()

    # ------------ Train config ----------------
    train_cfg = TrainConfig()
    log_dir = f"logs/{train_cfg.runner.experiment_name}"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = PPOEnv(
        [gs.morphs.Plane()],
        reward_functions,
        num_environments,
        sim_cfg,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        urdf_path,
    )

    runner = OnPolicyRunner(env, asdict(train_cfg), log_dir, device=gs.device)

    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


def cli_entrypoint():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the config directory",
        default=Path(__file__).resolve(),
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Specify device which you want to run PPO and simulation.",
        type=str,
        choices=["cpu", "gpu"],
        required=True,
    )
    parser.add_argument(
        "--num_environments", help="Number of environments", type=int, default=4096
    )
    parser.add_argument(
        "--urdf_path",
        help="Path to the URDF file",
        type=str,
        default="urdf/go2/urdf/go2.urdf",
    )
    args = parser.parse_args()
    train(
        device=args.device,
        config_directory=args.config,
        num_environments=args.num_environments,
        urdf_path=args.urdf_path,
    )


if __name__ == "__main__":
    cli_entrypoint()
