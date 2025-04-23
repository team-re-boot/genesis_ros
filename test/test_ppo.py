from genesis_ros.ppo import ppo_train
from genesis_ros.ppo import ppo_eval


def test_ppo_train():
    ppo_train.main()


def test_ppo_eval():
    ppo_eval.run_eval("genesis_ros_ppo", 100, 100, False)
