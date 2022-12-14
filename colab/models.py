import torch
import torch.nn as nn
import numpy as np


class AtariCNN(nn.Module):
  def __init__(self, state_dim, action_dim, epsilon_decay=1, exploit=False):
      super().__init__()
      self.conv = nn.Sequential(
          nn.Conv2d(state_dim, 16, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(16, 32, kernel_size=4, stride=2),
          nn.ReLU(),
      )
      self.fc = nn.Sequential(
          nn.Linear(2592, 256),
          nn.ReLU(),
          nn.Linear(256, action_dim)
      )
      self.epsilon_decay = epsilon_decay
      self.exploit = exploit

  def forward(self, x):
      x = torch.tensor(x, dtype=torch.float32)
      x = self.conv(x)
      x = torch.flatten(x, 1) # keep batch size
      x = self.fc(x)
      return x

  def get_action(self, actions, episode):
      if self.exploit:
        return np.argmax(actions.detach().cpu().numpy())

      epsilon = 0.01 + 0.99 * np.exp(-1 * episode / self.epsilon_decay)
      if np.random.rand() < epsilon:
          return np.random.choice(len(actions))
    
      return np.argmax(actions.detach().cpu().numpy())

  # input state -> get action sampled from current policy and q_value
  def act(self, state, episode=0):
      state = np.expand_dims(state, 0)
      q_values = self.forward(state).squeeze(0)
      action = self.get_action(q_values, episode)
      return action