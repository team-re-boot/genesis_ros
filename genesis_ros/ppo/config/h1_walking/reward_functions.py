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

    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    reward_functions.append((reward_similar_to_default, -0.5))

    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - 0.7)

    reward_functions.append((reward_base_height, -1.0))

    # def reward_terminate(self):
    #     # print(torch.sum(self.reset_buf, dim=0))
    #     return torch.where(self.reset_buf, 200.0, 0.0)

    # reward_functions.append((reward_terminate, -1.0))

    def reward_alive(self):
        return self.episode_length_buf / self.max_episode_length

    reward_functions.append((reward_alive, 1.0))

    def convex_blend_reward(
        base_reward: torch.Tensor,
        threshold: float = 0.4,
        beta: float = 0.03,  # さらに弱め
        p: float = 2.0,
    ) -> torch.Tensor:
        max_th = threshold
        base = base_reward.clamp(min=0.0, max=max_th)
        r = (base / max_th).clamp(0.0, 1.0)
        shaped = max_th * r.pow(p)
        blended = (1.0 - beta) * base + beta * shaped
        return blended.clamp(min=0.0, max=max_th)  # 出力安全域

    def reward_feet_air_time_from_phase(self):
        # --- 接地検出：Z成分を使い、ヒステリシスでチャタリング抑制 ---
        fz = self.contact_forces[:, :, 2]  # [N,2] 垂直
        hi, lo = 5.0, 3.0  # 例: N単位。環境に合わせて調整
        if not hasattr(self, "_foot_contact_state"):
            self._foot_contact_state = torch.zeros_like(fz, dtype=torch.bool)
        foot_contact = torch.where(self._foot_contact_state, fz > lo, fz > hi)
        # 次ステップ用に保持（勾配には載せない）
        self._foot_contact_state = foot_contact.detach()

        contact_count = foot_contact.int().sum(dim=1)  # [N]
        single_stance = contact_count == 1
        double_swing = contact_count == 0
        double_stance = contact_count == 2

        # --- 「空中進捗」：実接地に基づく遊脚だけを評価 ---
        swing_mask = ~foot_contact  # [N,2]
        air_progress = torch.where(
            swing_mask,
            (self.leg_phase - 0.55).clamp(min=0.0),
            torch.zeros_like(self.leg_phase),
        )  # [N,2]
        # 遊脚側のみの値を抽出（single時は1脚だけ非ゼロ）
        per_row = (air_progress * swing_mask.float()).sum(dim=1)  # [N]

        # 片足支持のときだけ正の基礎報酬
        base = torch.where(single_stance, per_row, torch.zeros_like(per_row))

        max_th = 0.4
        pos = convex_blend_reward(base, threshold=max_th, beta=0.03, p=2.0)

        # --- 明確な禁止項：二足同時に浮いたら強い罰 ---
        k0 = 0.4 * max_th  # 二足浮きペナルティの強さ（まずはこれくらいから）
        penalty_no_contact = -k0 * double_swing.float()

        # 二足接地は少しだけ不利（歩行を促すため、弱め）
        k2 = 0.1 * max_th
        penalty_double_stance = -k2 * double_stance.float()

        # “1本だけ接地”を中心に寄せる滑らかな形（任意：弱めの二乗罰）
        lam = 0.05 * max_th
        penalty_cc = -lam * (contact_count.float() - 1.0).pow(2)

        reward = pos + penalty_no_contact + penalty_double_stance + penalty_cc

        # 速度指令が小さいときはゼロ
        moving = (torch.norm(self.commands, dim=1) > 0.1).float()
        reward = reward * moving

        # 数値安定のために全体を最終クリップ（上下）
        reward = reward.clamp(min=-max_th, max=max_th)
        return reward

    reward_functions.append((reward_feet_air_time_from_phase, 1.5))

    def reward_feet_slide(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, :, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    reward_functions.append((reward_feet_slide, -0.25))

    return reward_functions
