import genesis as gs  # type: ignore
from genesis_ros.genesis_ros_env import GenesisRosEnv, genesis_entity
from genesis_ros.genesis_ros_env_options import (
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

    env = GenesisRosEnv(
        1,
        SimulationConfig(),
        EnvironmentConfig(),
        ObservationConfig(),
        RewardConfig(),
        CommandConfig(),
        "urdf/go2/urdf/go2.urdf",
    )
