import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
import numpy as np
import gymnasium as gym
from collections import deque
import random
import os
import matplotlib.pyplot as plt

# TD3 (Twin-Delayed DDPG)

# Hyperparameters

# Number of steps to train over
TOTAL_STEPS = 200000
# Number of steps to explore (sample from uniform random distribution)
EXPLORE_STEPS = 10000
# std for exploration noise
EXPLORE_STD = 0.5
# std for target policy smoothing from TD3 paper, clipped to [-0.5, 0.5]
SMOOTHING_STD = 0.2
# update actor and target networks every other step (critic updated every step)
POLICY_UPDATE = 2
# Steps per rollouts
BATCH_SIZE = 1024
# Mini-Batch size
MINI_SIZE = 32
# Replay Buffer Size
BUFFER_SIZE = 50000
# Discount factor
GAMMA = 0.99
# Polyak constant for updating target networks
POLYAK = 0.005
# Learning rate
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3

# model paths
PATH_ACTOR = './models/td3_actor.pth'
PATH_CRITIC1 = './models/td3_critic1.pth'
PATH_CRITIC2 = './models/td3_critic2.pth'

# A given S
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_high, action_low):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # probably should be using constant std or tanh for outputting action
        self.logstd = 0.01 * nn.Parameter(torch.ones(action_dim, dtype=torch.float32))
        self.action_high = action_high
        self.action_low = action_low

    def forward(self, x):
        mu = self.layers(x)
        std = torch.exp(self.logstd)

        return mu, std

    def get_action(self, state, exploit=False, smoothing=False):
        mu, std = self.forward(state)
        # sample noise from zero-mean Gaussians
        dist, explore_noise, smoothing_noise = Normal(mu, std), Normal(0, EXPLORE_STD), Normal(0, SMOOTHING_STD)
        if exploit:
            action = torch.clamp(dist.rsample(), self.action_low, self.action_high)
        else:
            # policy smoothing when choosing target action for update
            if smoothing:
                noise = torch.clamp(smoothing_noise.sample(), -0.5, 0.5)
                action = torch.clamp(dist.rsample() + noise, self.action_low, self.action_high)
            # gaussian noise for exploration
            else:
                action = torch.clamp(dist.rsample() + explore_noise.sample(), self.action_low, self.action_high)
        
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


class TD3Agent():
    def __init__(self, env_name, seed, render_mode='rgb_array'):
        # initialize env
        self.env = gym.make(env_name, render_mode=render_mode)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.action_low = torch.from_numpy(self.env.action_space.low)
        self.action_high = torch.from_numpy(self.env.action_space.high)

        # initialize networks
        self.actor = Actor(obs_dim, action_dim, self.action_high, self.action_low)
        self.critic1 = Critic(obs_dim, action_dim)
        self.critic2 = Critic(obs_dim, action_dim)

        self.load()
        
        self.target_actor = Actor(obs_dim, action_dim, self.action_high, self.action_low)
        self.target_critic1 = Critic(obs_dim, action_dim)
        self.target_critic2 = Critic(obs_dim, action_dim)

        # initialize target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=CRITIC_LR)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=CRITIC_LR)

        # initialize seeding
        self.set_global_seeds(seed)

    def set_global_seeds(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load(self, actor_path=PATH_ACTOR, critic1_path=PATH_CRITIC1, critic2_path=PATH_CRITIC2):
        if os.path.exists(actor_path) and os.path.exists(critic1_path) and os.path.exists(critic2_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic1.load_state_dict(torch.load(critic1_path))
            self.critic2.load_state_dict(torch.load(critic2_path))

    def save(self, actor_path=PATH_ACTOR, critic1_path=PATH_CRITIC1, critic2_path=PATH_CRITIC2):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
    
    def update_actor(self, qvalues):
        self.actor_optim.zero_grad()
        loss = -qvalues.mean()
        loss.backward(retain_graph=True)
        self.actor_optim.step()

    def update_critics(self, qvalues1, qvalues2, targets):
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        loss1 = (qvalues1 - targets).pow(2).mean()
        loss2 = (qvalues2 - targets).pow(2).mean()

        loss1.backward()
        loss2.backward()

        self.critic1_optim.step()
        self.critic2_optim.step()

    # soft-update for actor network
    def soft_update(self):
        with torch.no_grad():
            actor_param = list(self.actor.parameters())
            critic1_param = list(self.critic1.parameters())
            critic2_param = list(self.critic2.parameters())
            for i, p in enumerate(self.target_actor.parameters()):
                updated_p = (1 - POLYAK) * p + POLYAK * actor_param[i]
                p.copy_(updated_p)
            for i, p in enumerate(self.target_critic1.parameters()):
                updated_p = (1 - POLYAK) * p + POLYAK * critic1_param[i]
                p.copy_(updated_p)
            for i, p in enumerate(self.target_critic2.parameters()):
                updated_p = (1 - POLYAK) * p + POLYAK * critic2_param[i]
                p.copy_(updated_p)


def plot(mean_rewards, episode_rewards, global_step):
    plt.plot(mean_rewards)
    plt.title(f'Mean Rewards at Step {global_step}')
    plt.xlabel('Update')
    plt.ylabel('Reward')
    plt.savefig('./results/TD3_Pendulum_Mean')
    plt.clf()

    # plot episode rewards
    plt.plot(episode_rewards)
    plt.title(f'Episode Rewards at Step {global_step}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./results/TD3_Pendulum_Episode')
    plt.clf()
    

def learn(agent, buffer, step):
    s, a, r, next_s, done = buffer.sample(MINI_SIZE)
    s = torch.tensor(s)
    a = torch.tensor(a) 
    r = torch.tensor(r).unsqueeze(1)
    next_s = torch.tensor(next_s)
    done = torch.tensor(done).unsqueeze(1)

    target_a = agent.target_actor.get_action(next_s, smoothing=True)

    with torch.no_grad():
        min_qvalue = torch.min(agent.target_critic1(next_s, target_a), agent.target_critic2(next_s, target_a))
        targets = r + GAMMA * (1 - done) * min_qvalue

    qvalues1 = agent.critic1(s, a)
    qvalues2 = agent.critic2(s, a)
    
    agent.update_critics(qvalues1, qvalues2, targets)

    if step % POLICY_UPDATE == 0:
        agent.update_actor(agent.critic1(s, agent.actor.get_action(s, exploit=True)))
        agent.soft_update()



def train(agent):
    agent.actor.train()
    agent.critic1.train()
    agent.critic2.train()

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
                learn(agent, buffer, global_step)

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
        if mean_reward > -150:
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
    agent = TD3Agent(env_name, seed=10)
    # train(agent)
    eval(agent)




