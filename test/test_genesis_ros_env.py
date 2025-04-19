import genesis as gs  # type: ignore
from genesis_ros.ppo.ppo_env import PPOEnv, genesis_entity
from genesis_ros.ppo.ppo_env_options import (
    SimulationConfig,
    EnvironmentConfig,
    ObservationConfig,
    RewardConfig,
    CommandConfig,
)


def test_genesis_ros_env():
    gs.init(logging_level="warning", backend=gs.cpu)

    @genesis_entity
    def add_plane():
        return gs.morphs.Plane()

    env = PPOEnv(
        1,
        SimulationConfig(),
        EnvironmentConfig(),
        ObservationConfig(),
        RewardConfig(),
        CommandConfig(),
        "urdf/go2/urdf/go2.urdf",
    )
