import torch
from genesis.utils.geom import quat_to_R


def get_reward_functions():
    reward_functions = []

    # ------------ reward functions----------------
    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - 0.9)

    reward_functions.append((reward_base_height, -1.0))

    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    reward_functions.append((reward_lin_vel_z, -1.0))

    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    reward_functions.append((reward_action_rate, -0.5))

    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    reward_functions.append((reward_similar_to_default, -0.1))

    def reward_base_roll_pitch(self):
        # Penalize base roll and pitch angles
        pitch = self.base_euler[:, 1]
        roll = self.base_euler[:, 0]
        return 1 / torch.square(pitch) + torch.square(roll)

    reward_functions.append((reward_base_roll_pitch, 0.1))

    def reward_alive(self):
        return 1

    reward_functions.append((reward_alive, 1.0))

    return reward_functions
