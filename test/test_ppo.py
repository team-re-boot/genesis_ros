from genesis_ros.ppo import ppo_train, ppo_eval
from pathlib import Path


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


def test_ppo_eval():
    ppo_eval.eval(
        device="cpu",
        exp_name="go2_walking",
        ckpt=100,
        max_steps=100,
        show_viewer=False,
        urdf_path="urdf/go2/urdf/go2.urdf",
    )
