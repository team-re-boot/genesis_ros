import torch
from genesis.utils.geom import quat_to_R


def get_reward_functions():
    reward_functions = []

    # ------------ reward functions----------------
    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - 1.05)

    reward_functions.append((reward_base_height, -10.0))

    def reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / 0.25)

    reward_functions.append((reward_tracking_lin_vel, 1.0))

    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    reward_functions.append((reward_tracking_ang_vel, 0.5))

    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    reward_functions.append((reward_lin_vel_z, -2.0))

    def reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    reward_functions.append((reward_ang_vel_xy, -0.05))

    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    reward_functions.append((reward_action_rate, -0.01))

    def reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits_lower).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits_upper).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    reward_functions.append((reward_dof_pos_limits, -5.0))

    def reward_alive(self):
        return 1

    reward_functions.append((reward_alive, 0.15))

    def reward_contact(self):
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.env_cfg.foot_links)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, i, 2] > 1
            reward += ~(contact ^ is_stance)
        return reward

    reward_functions.append((reward_contact, 0.18))

    def reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, :, :3], dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    reward_functions.append((reward_feet_swing_height, -20.0))

    def reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, :, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    reward_functions.append((reward_contact_no_vel, -0.2))

    def reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    reward_functions.append((reward_orientation, -1.0))

    def reward_hip_pos(self):
        return torch.sum(torch.square(self.hip_dof_pos), dim=1)

    reward_functions.append((reward_hip_pos, -1.0))

    return reward_functions
