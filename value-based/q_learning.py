import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# Q-Learning (Off-Policy TD Control)

# Episodes
EPISODES = 2000
# Max Steps per Episode
MAX_STEPS = 1000
# Discount Factor
GAMMA = 0.99
# Learning Rate
LR = 1e-3
# Epsilon (e-greedy policy)
EPSILON = 0.5
# Seed
SEED = 11
# Solved Score
SOLVED_SCORE = 195
# Model path
PATH = './models/q_learning.pth'
# Environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# State-Action Value Network
class QNet(nn.Module):
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
        epsilon = 1 / (episode + 1)
        if np.random.rand() < epsilon:
            return np.random.choice(len(actions))
        
        return np.argmax(actions.detach().numpy())



def train():
    model = QNet(state_dim, action_dim)
    optim = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    model.train()

    total_rewards = []
    for episode in range(EPISODES):
        state = env.reset(seed=SEED)[0]
        rewards = []
        # run trajectory through episode
        for _ in range(MAX_STEPS):
            actions = model(state)
            action = model.get_action(actions, episode)

            next_s, r, done, _, _ = env.step(action)

            rewards.append(r)
            state = next_s

            # update network parameters
            next_actions = model(next_s)
            td_target = r + GAMMA * torch.max(next_actions)
            q_value = actions[action]

            optim.zero_grad()
            # MSE of TD-error
            loss = criterion(td_target, q_value)
            loss.backward()
            optim.step()

            if done:
                break

        total_rewards.append(np.sum(rewards))
        mean = np.mean(total_rewards[-100:])
        if episode % 100 == 0:
            print(f'EPISODE: {episode}, MEAN: {mean}')
        if mean > 195:
            print(f'Game Solved at Episode {episode}')
            break
    
    # plot results
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Q-Learning on CartPole')
    plt.xlim(right=2000)
    plt.ylim(top=500)
    reg = LinearRegression().fit(
        np.reshape(np.arange(len(total_rewards)), (-1, 1)),
        np.reshape(total_rewards, (-1, 1))
    )
    plt.plot(reg.predict(np.reshape(np.arange(len(total_rewards)), (-1, 1))))
    plt.show()

    torch.save(model.state_dict(), PATH)

def eval():
    model = QNet(state_dim, action_dim)
    model.eval()

    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))

    eval_episodes = 30
    eval_steps = 10000
    total_rewards = []
    for episode in range(eval_episodes):
        state = env.reset(seed=SEED)[0]
        rewards = []
        for _ in range(eval_steps):
            actions = model(state)
            action = np.argmax(actions.detach().numpy())

            next_s, r, done, _, _ = env.step(action)

            state = next_s
            rewards.append(r)
            if done:
                break
        
        total_rewards.append(np.sum(rewards))
        print(f'EPISODE: {episode}, REWARD: {np.sum(rewards)}')

    print(f'MEAN: {np.mean(total_rewards)}')
    


if __name__ == '__main__':
    train()
    eval()