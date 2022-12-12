from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from utils import get_evaluation_states, get_average_q_value, run_trials

# Deep Q-Learning Using DQN with Experience Replay

# Episodes
EPISODES = 2000
# Max Steps per Episode
MAX_STEPS = 1000
# Mini-Batch Size for Experience Replay
BATCH_SIZE = 32
# Size of Replay Buffer
BUFFER_SIZE = 1000
# Discount Factor
GAMMA = 0.99
# Learning Rate
LR = 1e-3
# Epsilon Decay -> Higher number decays slower y = 0.01 + 0.99e^(-x/EPSILON_DECAY)
EPSILON_DECAY = 400 # around 37.4% by episode 400
# Seed
SEED = 11
# Solved Score
SOLVED_SCORE = 195
# Model path
PATH = './models/deep_q_learning.pth'
# Environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# State-Action Value Network
class DQN(nn.Module):
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

# experience replay buffer
class ReplayBuffer():
    def __init__(self, max_length):
        self.buffer = deque(maxlen=max_length)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = zip(*experiences)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), done

    def __len__(self):
        return len(self.buffer)

# train from mini-batch of experiences
def train_off_experience(model, buffer, criterion, optim):
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.long)

    # get q values of actions taken
    q_values = model(state)
    q_value_from_action = torch.gather(q_values, 1, action.unsqueeze(1)).squeeze(1)

    # get best next q values: max(q_values)
    next_q_values = model(next_state)
    next_q_value_from_action = torch.max(next_q_values, 1)[0]

    # compute loss and update parameters
    optim.zero_grad()
    td_target = reward + (1 - done) * GAMMA * next_q_value_from_action
    td_loss = criterion(q_value_from_action, td_target)
    td_loss.backward()
    optim.step()


def train():
    model = DQN(state_dim, action_dim)
    optim = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    buffer = ReplayBuffer(max_length=BUFFER_SIZE)
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
            actions = model(state)
            action = model.get_action(actions, episode)

            next_s, r, done, _, _ = env.step(action)

            # add experience to replay buffer
            buffer.push(state, action, r, next_s, done)

            # sample from replay buffer if len(buffer) > BATCH_SIZE
            if len(buffer) > BATCH_SIZE:
                train_off_experience(model, buffer, criterion, optim)

            rewards.append(r)
            state = next_s
            
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
    
    torch.save(model.state_dict(), PATH)

    # return totals
    return total_rewards, total_q_values, solved_episode


if __name__ == '__main__':
    run_trials(5, train, 'Deep Q-Learning')