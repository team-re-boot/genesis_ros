import argparse
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
import pickle
import shutil
import os
from rsl_rl.runners import OnPolicyRunner
from dataclasses import asdict


def eval(
    exp_name: str,
    ckpt: int,
    max_steps: int = 100,
    show_viewer: bool = True,
    urdf_path: str = "urdf/go2/urdf/go2.urdf",
):
    gs.init(logging_level="warning", backend=gs.gpu)

    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{exp_name}/cfgs.pkl", "rb")
    )
    env = PPOEnv(
        [gs.morphs.Plane()],
        [],
        1,
        SimulationConfig(),
        env_cfg,
        obs_cfg,
        command_cfg,
        "urdf/go2/urdf/go2.urdf",
        show_viewer=show_viewer,
    )

    runner = OnPolicyRunner(env, asdict(train_cfg), log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    step = 0
    with torch.no_grad():
        while step < max_steps:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="genesis_ros_ppo")
    parser.add_argument(
        "-m", "--max_steps", type=int, default=100, help="Max steps to run"
    )
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument(
        "--urdf_path",
        help="Path to the URDF file",
        type=str,
        default="urdf/go2/urdf/go2.urdf",
    )
    args = parser.parse_args()
    eval(args.exp_name, args.ckpt, args.max_steps, args.urdf_path)


if __name__ == "__main__":
    main()
