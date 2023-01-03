import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
import numpy as np
import gymnasium as gym
from collections import deque
import random
import os
import matplotlib.pyplot as plt

# DDPG (Deep Deterministic Policy Gradient)

# Hyperparameters

# Number of steps to train over
TOTAL_STEPS = 200000
# Number of steps to explore (sample from uniform random distribution)
EXPLORE_STEPS = 5000
# Steps per rollouts
BATCH_SIZE = 1024
# Mini-Batch size
MINI_SIZE = 32
# Replay Buffer Size
BUFFER_SIZE = 50000
# Discount factor
GAMMA = 0.99
# Polyak constant for updating target networks
POLYAK = 0.001
# Learning rate
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3

# model paths
PATH_ACTOR = './models/ddpg_actor.pth'
PATH_CRITIC = './models/ddpg_critic.pth'

# A given S
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_high, action_low):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.logstd = 0.01 * nn.Parameter(torch.ones(action_dim, dtype=torch.float32))
        self.action_high = action_high
        self.action_low = action_low

    def forward(self, x):
        mu = self.layers(x)
        std = torch.exp(self.logstd)

        return mu, std

    def get_action(self, state, exploit=False):
        mu, std = self.forward(state)
        # sample noise from zero-mean Gaussian
        dist, noise = Normal(mu, std), Normal(0, 1)
        if exploit:
            action = torch.clamp(dist.sample(), self.action_low, self.action_high)
        else:
            action = torch.clamp(dist.rsample() + noise.sample(), self.action_low, self.action_high)
        
        return action

# Q(S,A) given S and A
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        qvalue = self.layers(input)
        return qvalue


# Experience Replay Buffer
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
        return np.concatenate(states), np.array(actions), np.array(rewards), np.concatenate(next_states), np.array(done)

    def __len__(self):
        return len(self.buffer)


class DDPGAgent():
    def __init__(self, env_name, seed, render_mode='rgb_array'):
        # initialize env
        self.env = gym.make(env_name, render_mode=render_mode)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.action_low = torch.from_numpy(self.env.action_space.low)
        self.action_high = torch.from_numpy(self.env.action_space.high)

        # initialize networks
        self.actor = Actor(obs_dim, action_dim, self.action_high, self.action_low)
        self.critic = Critic(obs_dim, action_dim)

        self.load()
        
        self.target_actor = Actor(obs_dim, action_dim, self.action_high, self.action_low)
        self.target_critic = Critic(obs_dim, action_dim)

        # initialize target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # initialize seeding
        self.set_global_seeds(seed)

    def set_global_seeds(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load(self, actor_path=PATH_ACTOR, critic_path=PATH_CRITIC):
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))

    def save(self, actor_path=PATH_ACTOR, critic_path=PATH_CRITIC):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def update_actor(self, qvalues):
        loss = -qvalues.mean()
        self.actor_optim.zero_grad()
        loss.backward(retain_graph=True)
        self.actor_optim.step()

    def update_critic(self, qvalues, targets):
        loss = (qvalues - targets).pow(2).mean()
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    # soft-update for target networks
    def soft_update(self):
        with torch.no_grad():
            actor_param = list(self.actor.parameters())
            critic_param = list(self.critic.parameters())
            for i, p in enumerate(self.target_actor.parameters()):
                updated_p = (1 - POLYAK) * p + POLYAK * actor_param[i]
                p.copy_(updated_p)
            for i, p in enumerate(self.target_critic.parameters()):
                updated_p = (1 - POLYAK) * p + POLYAK * critic_param[i]
                p.copy_(updated_p)


def plot(mean_rewards, episode_rewards, global_step):
    plt.plot(mean_rewards)
    plt.title(f'Mean Rewards at Step {global_step}')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.savefig('./results/DDPG_Pendulum')

    # plot episode rewards
    # plt.plot(episode_rewards)
    

def learn(agent, buffer):
    s, a, r, next_s, done = buffer.sample(MINI_SIZE)
    s = torch.tensor(s)
    a = torch.tensor(a)
    r = torch.tensor(r).unsqueeze(1)
    next_s = torch.tensor(next_s)
    done = torch.tensor(done).unsqueeze(1)

    target_a = agent.target_actor.get_action(next_s)
    target_qvalues = agent.target_critic(next_s, target_a).detach()

    targets = r + GAMMA * (1 - done) * target_qvalues
    qvalues = agent.critic(s, a)

    agent.update_critic(qvalues, targets)
    agent.update_actor(agent.critic(s, agent.actor.get_action(s)))

    agent.soft_update()


def train(agent):
    agent.actor.train()
    agent.critic.train()

    # initialize replay buffer
    buffer = ReplayBuffer(max_length=BUFFER_SIZE)

    global_step = 0
    num_updates = TOTAL_STEPS // BATCH_SIZE
    state = agent.env.reset()[0]
    total_mean_rewards = []
    episode_rewards = []
    for update in range(num_updates):
        rewards = []
        for step in range(BATCH_SIZE):
            global_step += 1

            with torch.no_grad():
                if global_step <= EXPLORE_STEPS:
                    action = Uniform(agent.action_low, agent.action_high).rsample()
                else:
                    action = agent.actor.get_action(torch.from_numpy(state))

            next_s, reward, terminated, truncated, info = agent.env.step(action)
            done = np.logical_or(terminated, truncated).astype(int)

            buffer.push(state, action.numpy(), reward.numpy(), next_s, done)

            # sample from buffer and train if enough samples
            if len(buffer) > BATCH_SIZE:
                learn(agent, buffer)

            state = next_s
            rewards.append(reward)
            if done:
                state = agent.env.reset()[0]
                episode_rewards.append(np.sum(rewards))
                rewards = []

        # plotting
        mean_reward = np.mean(episode_rewards[-100:])
        total_mean_rewards.append(mean_reward)
        print(f'Step: {global_step} Reward: {mean_reward:.3f}')
        plot(total_mean_rewards, episode_rewards, global_step)

        agent.save()

        # early stop for Pendulum-v1
        if mean_reward > -200:
            break
        

def eval(agent):
    state = agent.env.reset()[0]
    reward = []
    for _ in range(1000):
        action = agent.actor.get_action(torch.from_numpy(state), exploit=True)

        next_s, r, terminated, truncated, info = agent.env.step(action.detach().numpy())

        state = next_s
        reward.append(r)
        if terminated or truncated:
            state = agent.env.reset()[0]
            print(f'Reward: {np.sum(reward)}')
            reward = []




if __name__ == '__main__':
    env_name = 'Pendulum-v1'
    agent = DDPGAgent(env_name, seed=10, render_mode='human')
    # train(agent)
    eval(agent)




