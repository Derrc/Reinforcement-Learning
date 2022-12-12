import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from utils import get_evaluation_states, get_average_q_value, run_trials

# SARSA (On-Policy TD Control)

# Episodes
EPISODES = 2000
# Max Steps per Episode
MAX_STEPS = 1000
# Discount Factor
GAMMA = 0.99
# Learning Rate
LR = 1e-3
# Epsilon Decay
EPSILON_DECAY = 400 # around 37.4% by episode 400
# Seeding
SEED = 11
# Solved Score
SOLVED_SCORE = 195
# Model path
PATH = './models/sarsa.pth'
# Environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# State-Action Value Network
class SARSA(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.layers(x)
        return x

    def get_action(self, actions, episode):
        epsilon = 0.01 + 0.99 * np.exp(-1 * episode / EPSILON_DECAY)
        if np.random.rand() < epsilon:
            return np.random.choice(len(actions))
        
        return np.argmax(actions.detach().numpy())

    # input state -> get action sampled from current policy and q_value
    def act(self, state, episode):
        q_values = self.forward(state)
        action = self.get_action(q_values, episode)
        return action, q_values[action]



def train():
    model = SARSA(state_dim, action_dim)
    optim = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    model.train()

    # keep track of q-values in evaluation states
    evaluation_states = get_evaluation_states(env, 5)
    total_rewards = []
    total_q_values = []
    solved_episode = 0
    for episode in range(EPISODES):
        state = env.reset(seed=SEED)[0]
        rewards = []
        # run trajectory through episode
        for _ in range(MAX_STEPS):
            # get action sampled from policy and q_value
            action, q_value = model.act(state, episode)

            next_s, r, done, _, _ = env.step(action)

            rewards.append(r)
            state = next_s

            # update network parameters
            # get next q_value sampled from policy
            next_q_value = model.act(next_s, episode)[1]
            td_target = r + GAMMA * next_q_value

            optim.zero_grad()
            # MSE of TD-error
            loss = criterion(td_target, q_value)
            loss.backward()
            optim.step()

            if done:
                break

        total_rewards.append(np.sum(rewards))
        total_q_values.append(get_average_q_value(model, evaluation_states))
        mean = np.mean(total_rewards[-100:])
        if episode % 100 == 0:
            print(f'EPISODE: {episode}, MEAN: {mean}')
        if mean > 195:
            solved_episode = episode
            print(f'Game Solved at Episode {episode}')
            break
    
    # save model
    torch.save(model.state_dict(), PATH)

    return total_rewards, total_q_values, solved_episode


if __name__ == '__main__':
    run_trials(5, train, 'SARSA')