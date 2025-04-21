from genesis_ros.ppo.ppo_env import (
    PPOEnv,
    genesis_entity,
    set_reward_scale,
    ppo_reward_function,
)
import genesis as gs  # type: ignore
import torch
from genesis_ros.ppo.ppo_env_options import (
    SimulationConfig,
    EnvironmentConfig,
    ObservationConfig,
    RewardConfig,
    CommandConfig,
)

if __name__ == "__main__":
    gs.init(logging_level="warning", backend=gs.cpu)

    @genesis_entity
    def add_plane():
        return gs.morphs.Plane()

    # ------------ reward functions----------------
    @ppo_reward_function
    @set_reward_scale(1.0)
    def reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg.tracking_sigma)

    @ppo_reward_function
    @set_reward_scale(0.2)
    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg.tracking_sigma)

    @ppo_reward_function
    @set_reward_scale(-1.0)
    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    @ppo_reward_function
    @set_reward_scale(-0.005)
    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    @ppo_reward_function
    @set_reward_scale(-0.1)
    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    @ppo_reward_function
    @set_reward_scale(-50.0)
    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg.base_height_target)

    env = PPOEnv(
        1,
        SimulationConfig(),
        EnvironmentConfig(),
        ObservationConfig(),
        RewardConfig(),
        CommandConfig(),
        "urdf/go2/urdf/go2.urdf",
    )
