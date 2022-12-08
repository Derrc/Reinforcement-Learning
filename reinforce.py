import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# REINFORCE Policy Gradient Algorithm

# Episodes
EPISODES = 1000
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
PATH = './models/reinforce.pth'
# Environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

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


def get_cumulative_rewards(rewards):
    cr = [rewards[-1]]
    for i in range(len(rewards)-2, -1, -1):
        cr.append(rewards[i] + GAMMA * cr[-1])
    cr.reverse()
    return cr

def train():
    model = Policy(state_dim, action_dim)
    optim = torch.optim.SGD(model.parameters(), lr=LR)
    model.train()

    total_rewards = []
    for episode in range(EPISODES):
        state = env.reset(seed=SEED)[0]
        log_probs = []
        rewards = []
        # run trajectory through episode
        for _ in range(MAX_STEPS):
            action, log_prob = model(state)

            next_s, r, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(r)

            state = next_s
            if done:
                break

        # update policy parameters
        optim.zero_grad()
        cumulative_rewards = torch.tensor(get_cumulative_rewards(rewards))
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * cumulative_rewards).mean()
        policy_loss.backward()
        optim.step()

        total_rewards.append(np.sum(rewards))
        mean = np.mean(total_rewards[-100:])
        if episode % 100 == 0:
            print(f'EPISODE: {episode}, MEAN: {mean}')
        if mean > 195:
            print('Game Solved!')

    # plot results
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('REINFORCE Policy Gradient on CartPole')
    reg = LinearRegression().fit(
        np.reshape(np.arange(len(total_rewards)), (-1, 1)),
        np.reshape(total_rewards, (-1, 1))
    )
    plt.plot(reg.predict(np.reshape(np.arange(len(total_rewards)), (-1, 1))))
    plt.show()

    torch.save(model.state_dict(), PATH)

def eval():
    model = Policy(state_dim, action_dim)
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
            action = model(state)[0]

            next_s, r, done, _, _ = env.step(action)

            state = next_s
            rewards.append(r)
            if done:
                break
        
        total_rewards.append(np.sum(rewards))
        print(f'EPISODE: {episode}, REWARD: {np.sum(rewards)}')

    print(f'MEAN: {np.mean(total_rewards)}')
    


if __name__ == '__main__':
    # train()
    eval()




