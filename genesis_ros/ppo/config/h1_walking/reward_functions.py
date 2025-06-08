import torch
from genesis.utils.geom import quat_to_R


def get_reward_functions():
    reward_functions = []

    # ------------ reward functions----------------
    # def reward_base_height(self):
    #     # Penalize base height away from target
    #     return torch.square(self.base_pos[:, 2] - 1.0)

    # reward_functions.append((reward_base_height, -100.0))

    def penalty_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - 0.8)

    reward_functions.append((penalty_base_height, -1.0))

    # def reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(
    #         torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
    #     )
    #     return torch.exp(-lin_vel_error / 0.25 + penalty_base_height(self))

    # reward_functions.append((reward_tracking_lin_vel, 10.0))

    # def reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error / 0.25 + penalty_base_height(self))

    # reward_functions.append((reward_tracking_ang_vel, 0.2))

    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    reward_functions.append((reward_lin_vel_z, -1.0))

    # def reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # reward_functions.append((reward_action_rate, -0.5))

    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    reward_functions.append((reward_similar_to_default, -0.1))

    return reward_functions
