from rsl_rl.modules.actor_critic import ActorCritic
import torch.nn as nn
import torch


class ActorPolicy(nn.Module):
    def __init__(self, actor_critic: ActorCritic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_critic.act_inference(observations)
