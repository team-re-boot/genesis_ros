import torch


def get_reward_functions():
    reward_functions = []

    # ------------ reward functions----------------
    def reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(
        #     torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        # )
        lin_vel_error = self.commands[:, 0] - self.base_lin_vel[:, 0]
        return torch.exp(-lin_vel_error / 0.25)
        # scaling_factorは報酬の鋭敏さを調整するハイパーパラメータ。
        # この値が大きいほど、少しの誤差でも報酬が急激に減少する。
        # scaling_factor = 5.0
        # return torch.exp(
        #     -scaling_factor
        #     * torch.sum(
        #         torch.square(
        #             torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        #         ),
        #         dim=1,
        #     )
        # )

    reward_functions.append((reward_tracking_lin_vel, 1.0))

    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    reward_functions.append((reward_tracking_ang_vel, 0.2))

    def reward_lin_vel_z(self):
        """
        Z軸方向の線形速度が0に近いほど高い報酬を与える。
        torch.exp(-a * x^2) の形をしており、
        速度(x)が0のときに最大値1.0をとり、速度が大きくなるにつれて0に近づく。
        """
        # scaling_factorは報酬の鋭敏さを調整するハイパーパラメータ。
        # この値が大きいほど、少しの上下動でも報酬が急激に減少する。
        scaling_factor = 5.0
        return torch.exp(-scaling_factor * torch.square(self.base_lin_vel[:, 2]))

    reward_functions.append((reward_lin_vel_z, 1.0))

    def reward_action_rate(self):
        # Penalize changes in actions
        # return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        # scaling_factorは報酬の鋭敏さを調整するハイパーパラメータ。
        # この値が大きいほど、少しの誤差でも報酬が急激に減少する。
        scaling_factor = 15.0
        return torch.exp(
            -scaling_factor
            * torch.sum(
                torch.square(self.last_actions - self.actions),
                dim=1,
            )
        )

    reward_functions.append((reward_action_rate, 1.0))

    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    reward_functions.append((reward_similar_to_default, -1.5))

    def reward_base_height(self):
        # この値が大きいほど、少しの誤差でも報酬が急激に減少する。
        scaling_factor = 5.0
        return torch.exp(-scaling_factor * torch.abs(self.base_pos[:, 2] - 0.7))

        # Penalize base height away from target
        # return torch.abs(self.base_pos[:, 2] - 0.7)

    reward_functions.append((reward_base_height, 1.0))

    # def reward_terminate(self):
    #     # print(torch.sum(self.reset_buf, dim=0))
    #     return torch.where(self.reset_buf, 200.0, 0.0)

    # reward_functions.append((reward_terminate, -1.0))

    # def reward_alive(self):
    #     return self.episode_length_buf / self.max_episode_length

    # reward_functions.append((reward_alive, 1.0))

    # 補助関数: 0〜1に滑らかにマップ
    def smoothstep(x, low, high):
        t = ((x - low) / (high - low)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # 線形(base)と凸(shaped)をブレンド（パラメータは定数）
    def convex_blend_reward(
        base_reward: torch.Tensor, threshold: float, beta: float, p: float
    ) -> torch.Tensor:
        base = base_reward.clamp(min=0.0, max=threshold)
        r = (base / threshold).clamp(0.0, 1.0)
        shaped = threshold * r.pow(p)
        blended = (1.0 - beta) * base + beta * shaped
        return blended.clamp(min=0.0, max=threshold)

    def reward_feet_air_time_stable(self):
        # ================= 定数（ここだけ調整すればOK） =================
        MAX_TH = 0.40  # 報酬上限
        CONTACT_SPLIT = 0.55  # 接地/遊脚の位相しきい
        CMD_MIN = 0.10  # 動作している判定
        BETA = 0.30  # 線形:凸 のブレンド率（凸の効き）
        P = 2.50  # 凸の強さ（>1で上側を強調）
        UPR_LOW = 0.85  # 直立ゲートの下限
        UPR_HIGH = 0.97  # 直立ゲートの上限
        W_ALIVE = 0.05 * MAX_TH  # 直立ボーナス
        W_OMEGA = 0.05 * MAX_TH  # 角速度の抑制ボーナス
        W_TAU = 0.02 * MAX_TH  # トルクの抑制ボーナス
        OMEGA_SCALE = 4.0  # 角速度のスケール
        TAU_SCALE = 2.0  # トルクのスケール
        # ============================================================

        leg_phase = self.leg_phase  # [N,2]
        device = leg_phase.device
        N = leg_phase.size(0)

        # 位相ベースの接地近似（実接地があるなら後段で置き換え）
        in_contact = leg_phase <= CONTACT_SPLIT  # [N,2] bool
        single_stance = in_contact.int().sum(dim=1) == 1

        # 遊脚側の空中進捗（片足支持のときだけ）
        swing_mask = ~in_contact
        air_progress = torch.where(
            swing_mask,
            (leg_phase - CONTACT_SPLIT).clamp(min=0.0),
            torch.zeros_like(leg_phase),
        )  # [N,2]
        base_linear = torch.where(
            single_stance, air_progress.max(dim=1).values, torch.zeros(N, device=device)
        )  # [N]

        # 指令が小さいときは0
        moving = (torch.norm(self.commands, dim=1) > CMD_MIN).float()
        base_linear = base_linear * moving

        # 直立ゲート
        q = self.base_quat / torch.linalg.norm(
            self.base_quat, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        w, x, y, z = q.unbind(-1)
        up_z = 1 - 2 * (x * x + y * y)
        g_upright = smoothstep(up_z, UPR_LOW, UPR_HIGH)  # [0,1]

        # 接地ゲート：少なくとも1脚接地
        # 実接地フラグ: 例) 法線力 > ϵ
        # foot_contact: [N,2] bool
        foot_contact = self.contact_forces[:, :, 2] > 1.0
        contact_count = foot_contact.int().sum(dim=1)  # [N] 0,1,2
        stance_ok = (contact_count >= 1).float()  # 少なくとも1脚
        flight = (contact_count == 0).float()  # 両足離地

        gate = g_upright * stance_ok  # [0,1]

        # 凸ブレンド（定数パラメータ）
        shaped = convex_blend_reward(base_linear, threshold=MAX_TH, beta=BETA, p=P)

        W_FLIGHT = 0.1 * MAX_TH  # 離地ペナルティの強さ（要調整）

        # 安全ゲート適用
        gate = g_upright * stance_ok
        reward = gate * shaped

        # 生存ボーナスは“接地時のみ”付与（空中では0）
        reward = reward + W_ALIVE * g_upright * stance_ok

        # 両足離地は明示的に減点
        reward = reward - W_FLIGHT * flight

        return reward.clamp(0.0, MAX_TH)

    reward_functions.append((reward_feet_air_time_stable, 1.5))

    def reward_feet_slide(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, :, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    reward_functions.append((reward_feet_slide, -0.25))

    def reward_posture(self):
        pitch_sigma: float = 0.10  # 許容幅 (rad)
        q = self.base_quat / torch.linalg.norm(
            self.base_quat, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        w, x, y, z = q.unbind(-1)
        # 回転後の (0,0,1) を直接計算（回転行列の第3列に相当）
        up_x = 2 * (x * z + w * y)
        up_y = 2 * (y * z - w * x)
        up_z = 1 - 2 * (x * x + y * y)
        base_up = torch.stack([up_x, up_y, up_z], dim=-1)
        # 符号なしの傾き：直立(0)からの角度
        up_z = base_up[:, 2].clamp(-1.0, 1.0)
        tilt = torch.acos(up_z)  # 0=直立, 増えるほど傾きが大きい
        r_posture = torch.exp(-((tilt / pitch_sigma) ** 2))
        return r_posture.clamp(0.0, 1.0)

    reward_functions.append((reward_posture, 2.0))

    return reward_functions
