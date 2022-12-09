import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# REINFORCE Policy Gradient Algorithm

# Episodes
EPISODES = 2000
# Max Steps per Episode
MAX_STEPS = 1000
# Discount Factor
GAMMA = 0.99
# Learning Rate
LR = 1e-3
# Seed
SEED = 11
# Solved Score
SOLVED_SCORE = 195
# Model path
PATH = './models/reinforce_baseline.pth'
# Environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# State-Value Network as Baseline -> V(s)
class Baseline(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.layers(x)
        return x.squeeze()

# Policy Network: S -> pi(A|S)
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.layers(x)
        actions = torch.softmax(x, dim=0)
        action = self.get_action(actions)
        log_prob = torch.log(actions)[action]
        return action, log_prob

    def get_action(self, actions):
        return np.random.choice(self.action_dim, p=actions.detach().numpy())

# returns cumulative rewards throughout sampled episode
def get_cumulative_rewards(rewards):
    cr = [rewards[-1]]
    for i in range(len(rewards)-2, -1, -1):
        cr.append(rewards[i] + GAMMA * cr[-1])
    cr.reverse()
    return cr

def train():
    policy = Policy(state_dim, action_dim)
    baseline = Baseline(state_dim)
    policy_optim = torch.optim.SGD(policy.parameters(), lr=LR)
    baseline_optim = torch.optim.SGD(baseline.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    policy.train()
    baseline.train()

    total_rewards = []
    for episode in range(EPISODES):
        state = env.reset(seed=SEED)[0]
        log_probs = []
        rewards = []
        values = []
        # run trajectory through episode
        for _ in range(MAX_STEPS):
            action, log_prob = policy(state)
            value = baseline(state)

            next_s, r, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(r)
            values.append(value)

            state = next_s
            if done:
                break

        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        cumulative_rewards = torch.tensor(get_cumulative_rewards(rewards))

        # update baseline parameters
        baseline_optim.zero_grad()
        baseline_loss = criterion(values, cumulative_rewards)
        baseline_loss.backward()
        baseline_optim.step()

        # update policy parameters
        policy_optim.zero_grad()
        policy_loss = -(log_probs * (cumulative_rewards - values).detach()).mean()
        policy_loss.backward()
        policy_optim.step()

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
    plt.title('REINFORCE with Baseline Policy Gradient on CartPole')
    plt.xlim(right=2000)
    plt.ylim(top=500)
    reg = LinearRegression().fit(
        np.reshape(np.arange(len(total_rewards)), (-1, 1)),
        np.reshape(total_rewards, (-1, 1))
    )
    plt.plot(reg.predict(np.reshape(np.arange(len(total_rewards)), (-1, 1))))
    plt.show()

    torch.save(policy.state_dict(), PATH)

def eval():
    policy = Policy(state_dim, action_dim)
    policy.eval()

    if os.path.exists(PATH):
        policy.load_state_dict(torch.load(PATH))

    eval_episodes = 30
    eval_steps = 10000
    total_rewards = []
    for episode in range(eval_episodes):
        state = env.reset(seed=SEED)[0]
        rewards = []
        for _ in range(eval_steps):
            action = policy(state)[0]

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




