import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Categorical

# Actor-Critic Models used for PPO
# TODO: Expand for multi-dimensional action space and TRPO

# Value Network: V(s)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# Policy Network: pi(a|s)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous, action_low=-1, action_high=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),                
            nn.Linear(128, action_dim)
        )
        self.action_dim = action_dim
        self.continuous = continuous
        if self.continuous:
            self.action_low = action_low
            self.action_high = action_high
            self.logstd = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, x):
        x = self.layers(x)
        if self.continuous:
            std = torch.exp(self.logstd)
            # mean, std
            return x, std
        x = torch.softmax(x, dim=1)
        return x

    def get_log_probs(self, state, actions):
        if self.continuous:
            mu, std = self.forward(state)
            dist = Normal(mu, std)
        else:
            curr_actions = self.forward(state)
            dist = Categorical(curr_actions)
        log_prob = dist.log_prob(actions)

        return log_prob

    def get_action(self,state):
        state = torch.from_numpy(state)
        if self.continuous:
            mu, std = self.forward(state)
            dist =  Normal(mu, std)
            action = torch.clamp(dist.sample(), self.action_low, self.action_high)
        else:
            actions = torch.softmax(self.layers(state), dim=0)
            dist = Categorical(actions)
            action = dist.sample()

        log_prob_action = dist.log_prob(action)
        action = action if self.continuous else action.item()
        return action, log_prob_action