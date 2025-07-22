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
        return torch.nn.functional.sigmoid(
            1 / (torch.square(pitch) + torch.square(roll))
        )

    reward_functions.append((reward_base_roll_pitch, 0.1))

    def reward_alive(self):
        return 1

    reward_functions.append((reward_alive, 1.0))

    def reward_contact(self):
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.env_cfg.foot_links)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, i, 2] > 1
            reward += ~(contact ^ is_stance)
        return reward

    reward_functions.append((reward_contact, 1.0))

    def reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, :, :3], dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    reward_functions.append((reward_feet_swing_height, -0.1))

    return reward_functions
