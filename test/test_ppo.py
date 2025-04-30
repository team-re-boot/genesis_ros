from genesis_ros.ppo import ppo_train
from pathlib import Path

# from genesis_ros.ppo import ppo_eval


def test_ppo_train():
    ppo_train.train(
        device="cpu",
        num_environments=1,
        urdf_path="urdf/go2/urdf/go2.urdf",
        config_directory=Path(__file__).resolve().parent
        / ".."
        / "genesis_ros"
        / "ppo"
        / "config"
        / "go2_walking",
    )


# def test_ppo_eval():
#     ppo_eval.run_eval("genesis_ros_ppo", 100, 100, False)
