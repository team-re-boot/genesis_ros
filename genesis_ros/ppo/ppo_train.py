from genesis_ros.ppo.ppo_env import PPOEnv
import genesis as gs  # type: ignore
import torch
from genesis_ros.ppo.ppo_env_options import (
    SimulationConfig,
    EnvironmentConfig,
    ObservationConfig,
    CommandConfig,
)
from genesis_ros.ppo.ppo_train_options import TrainConfig, Algorithm, Policy, Runner
from genesis_ros.ppo.actor_policy import ActorPolicy
from genesis_ros.util import call_function_in_another_file
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

    env_cfg = EnvironmentConfig.safe_load(config_directory / "environment_config.yaml")
    sim_cfg = SimulationConfig.safe_load(config_directory / "simulation_config.yaml")
    obs_cfg = ObservationConfig.safe_load(config_directory / "observation_config.yaml")
    command_cfg = CommandConfig.safe_load(config_directory / "command_config.yaml")

    if (config_directory / "reward_functions.py").exists():
        reward_functions = call_function_in_another_file(
            config_directory / "reward_functions.py", "get_reward_functions"
        )
    else:
        raise FileNotFoundError(
            f"reward_functions.py not found in {config_directory}. Please provide the correct path."
        )

    if (config_directory / "entities.py").exists():
        entities = call_function_in_another_file(
            config_directory / "entities.py", "get_entities"
        )
    else:
        raise FileNotFoundError(
            f"entities.py not found in {config_directory}. Please provide the correct path."
        )

    # ------------ Train config ----------------
    train_cfg = TrainConfig.safe_load(
        config_directory / "train_config.yaml",
    )
    log_dir = f"logs/{train_cfg.runner.experiment_name}"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, command_cfg, train_cfg, entities],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = PPOEnv(
        entities=entities,
        reward_functions=reward_functions,
        num_envs=num_environments,
        simulation_cfg=sim_cfg,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        command_cfg=command_cfg,
        urdf_path=urdf_path,
    )

    runner = OnPolicyRunner(env, asdict(train_cfg), log_dir, device=gs.device)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )
    torch.jit.trace(
        ActorPolicy(runner.alg.actor_critic),
        torch.zeros(env.num_actions * 3 + 3 * 3).to(device=gs.device),
    ).save(Path(log_dir) / "actor.pt")
    gs.destroy()


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
