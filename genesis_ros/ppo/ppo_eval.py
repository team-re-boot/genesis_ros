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
from genesis_ros.ros2_interface import builtin_interfaces, rosgraph_msgs, torch_msgs
from genesis_ros.ros2_interface import ROS2Interface
from genesis_ros.ros2_interface import torch_msgs
from genesis_ros.topic_interfaces import TopicInterface, NopInterface
import pickle
import shutil
import os
from rsl_rl.runners import OnPolicyRunner
from dataclasses import asdict
from typing import Union
import zenoh
import time


def eval(
    exp_name: str,
    ckpt: int,
    show_viewer: bool = True,
    urdf_path: str = "urdf/go2/urdf/go2.urdf",
    device: str = "gpu",
    interface: str = "python",
):
    if device == "cpu":
        gs.init(logging_level="warning", backend=gs.cpu)
    elif device == "gpu":
        gs.init(logging_level="warning", backend=gs.gpu)
    else:
        raise ValueError("Invalid device specified. Choose 'cpu' or 'gpu'.")

    topic_interface: TopicInterface
    if interface == "ros2":
        topic_interface = ROS2Interface(zenoh_config=zenoh.Config())
    if interface == "python":
        topic_interface = NopInterface()

    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, command_cfg, train_cfg, entities = pickle.load(
        open(f"logs/{exp_name}/cfgs.pkl", "rb")
    )
    env = PPOEnv(
        entities=entities,
        reward_functions=[],
        num_envs=1,
        simulation_cfg=SimulationConfig(),
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        command_cfg=command_cfg,
        urdf_path=urdf_path,
        show_viewer=show_viewer,
    )

    runner = OnPolicyRunner(env, asdict(train_cfg), log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    step = 0
    with torch.no_grad():
        topic_interface.subscribe("control/action", torch_msgs.msg.FP32Tensor)
        while True:
            sec = step * env.dt
            topic_interface.publish(
                "clock",
                rosgraph_msgs.msg.Clock(
                    clock=builtin_interfaces.msg.Time(
                        sec=int(sec), nanosec=int((sec - int(sec)) * 1e9)
                    )
                ),
            )
            topic_interface.publish(
                "control/observation", torch_msgs.msg.from_torch_tensor(obs)
            )
            if type(topic_interface) == NopInterface:
                actions = policy(obs)
            elif type(topic_interface) == ROS2Interface:
                topic_interface.spin()
                actions = topic_interface.spin_until_subscribe_new_data(
                    "control/action"
                ).to_torch_tensor()
            obs, rews, dones, infos = env.step(actions)
            if dones[0]:
                break
            step += 1
            time.sleep(0.1)
    gs.destroy()


def cli_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        help="Specify device which you want to run PPO and simulation.",
        type=str,
        choices=["cpu", "gpu"],
        required=True,
    )
    parser.add_argument("-e", "--exp_name", type=str, default="genesis_ros_ppo")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument(
        "--urdf_path",
        help="Path to the URDF file",
        type=str,
        default="urdf/go2/urdf/go2.urdf",
    )
    parser.add_argument(
        "-i",
        "--interface",
        help="message passing interface",
        type=str,
        default="python",
        choices=["python", "ros2"],
    )
    args = parser.parse_args()
    eval(
        exp_name=args.exp_name,
        ckpt=args.ckpt,
        urdf_path=args.urdf_path,
        show_viewer=True,
        device=args.device,
        interface=args.interface,
    )


if __name__ == "__main__":
    cli_entrypoint()
