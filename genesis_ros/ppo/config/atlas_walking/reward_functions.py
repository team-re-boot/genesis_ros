import torch


def get_reward_functions():
    reward_functions = []

    # ------------ reward functions----------------
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

    reward_functions.append((reward_tracking_ang_vel, 0.2))

    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    reward_functions.append((reward_lin_vel_z, -1.0))

    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    reward_functions.append((reward_action_rate, -0.005))

    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - 0.8)

    reward_functions.append((reward_base_height, -1.0))

    def reward_base_stability(self):
        # Penalize pitch / roll angle away from zero
        pitch_penalty = torch.square(self.base_euler[:, 1])
        roll_penalty = torch.square(self.base_euler[:, 0])
        return pitch_penalty * 5.0 + roll_penalty * 1.0
        # return pitch_penalty

    reward_functions.append((reward_base_stability, -0.01))

    def reward_excessive_contacts(self):
        left_foot_contact = self.robot.get_link("l_foot").get_pos()[:, 2] < 0.07
        left_foot_flat = (
            self.robot.get_link("l_foot").get_ang()[:, 1] < 0.17453292519943
        )  # 10 degrees in radians
        right_foot_contact = self.robot.get_link("r_foot").get_pos()[:, 2] < 0.07
        right_foot_flat = (
            self.robot.get_link("r_foot").get_ang()[:, 1] < 0.17453292519943
        )  # 10 degrees in radians

        left_foot_is_supported = (
            left_foot_contact & left_foot_flat & (~right_foot_contact)
        )
        right_foot_is_supported = (
            right_foot_contact & right_foot_flat & (~left_foot_contact)
        )

        num_supported = left_foot_is_supported.float() + right_foot_is_supported.float()
        reward = torch.where(
            num_supported == 1.0,
            torch.tensor(1.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )
        return reward

    reward_functions.append((reward_excessive_contacts, 100.0))

    return reward_functions
