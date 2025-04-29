from genesis_ros.ppo import ppo_train

# from genesis_ros.ppo import ppo_eval


def test_ppo_train():
    ppo_train.train(
        device="cpu", num_environments=1, urdf_path="urdf/go2/urdf/go2.urdf"
    )


# def test_ppo_eval():
#     ppo_eval.run_eval("genesis_ros_ppo", 100, 100, False)
