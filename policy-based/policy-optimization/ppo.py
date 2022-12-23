import torch
import torch.nn as nn
import numpy as np
from models import Critic, Actor
from utils import gae
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import os

# Proximal Policy Optimization

class PPOAgent():
    def __init__(self, env_name, continuous, **hyperparameters):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]

        self.init_hyperparameters(hyperparameters)

        if continuous:
            self.action_dim = self.env.action_space.shape[0]
            self.action_low = self.env.action_space.low[0]
            self.action_high = self.env.action_space.high[0]
            self.actor = Actor(self.obs_dim, self.action_dim, continuous, self.action_low, self.action_high).to(self.device)
        else:
            self.action_dim = self.env.action_space.n
            self.actor = Actor(self.obs_dim, self.action_dim, continuous)

        self.critic = Critic(self.obs_dim).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-6)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.actor_path = ''
        self.critic_path = ''


    def init_hyperparameters(self, hyperparameters):
        self.total_steps = 2000000
        self.batch_size = 2048
        self.minibatch_size = 32
        self.discount = 0.99
        self.entropy_coef = 0.01
        self.policy_clip = 0.2
        self.kl_target = 0.01
        self.max_grad_norm = 0.5
        self.gae_param = 0.95
        self.optim_epochs = 10
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.seed = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    def load(self):
        self.actor.load_state_dict(torch.load(self.actor_path))
        self.critic.load_state_dict(torch.load(self.critic_path))

    def save(self):
        torch.save(self.actor.state_dict(), self.actor_path)
        torch.save(self.critic.state_dict(), self.critic_path)


def clipped_loss(ratio, advantages, policy_clip):
    # no_clip = torch.clamp(ratio * advantages, min=1e3, max=None)
    no_clip = ratio * advantages
    clip = torch.clamp(ratio, 1 - policy_clip, 1 + policy_clip) * advantages
    return torch.min(no_clip, clip).mean()

def plot(global_step, rewards):
    plt.plot(rewards)
    plt.title(f'Rewards at Step {global_step}')
    plt.show()


def train(env_name, current_update, continuous):
    agent = PPOAgent(env_name, continuous, seed=10)

    b_size = agent.batch_size
    device = agent.device
    env = agent.env

    # load models if exists
    if os.path.exists(agent.actor_path):
        agent.load()

    # storage tensors
    obs = torch.zeros((b_size, agent.obs_dim)).to(device)
    next_obs = torch.zeros((b_size, agent.obs_dim)).to(device)
    actions = torch.zeros((b_size, agent.action_dim)).to(device) if continuous else torch.zeros(b_size).to(device)
    log_probs = torch.zeros(b_size).to(device)
    rewards = torch.zeros(b_size).to(device)
    dones = torch.zeros(b_size).to(device)

    global_step = 0
    total_rewards = []
    num_updates = agent.total_steps // b_size
    num_minibatches = b_size // agent.minibatch_size
    ob = torch.tensor(env.reset()[0], dtype=torch.float32)

    for update in range(current_update, num_updates):
        # anneal lr of optimizers
        rate = 1.0 - (update - 1.0) / num_updates
        lr = rate * agent.actor_lr
        agent.actor_optim.param_groups[0]['lr'] = lr
        agent.critic_optim.param_groups[0]['lr'] = lr

        # run trajectories -> generate batch
        episodes = 0
        for step in range(b_size):
            global_step += 1

            with torch.no_grad():
                action, log_prob = agent.actor.get_action(ob)

            next_ob, reward, terminated, truncated, info = env.step(action)
            next_ob = torch.tensor(next_ob, dtype=torch.float32)

            obs[step] = ob
            next_obs[step] = next_ob
            actions[step] = action
            log_probs[step] = log_prob
            rewards[step] = reward
            dones[step] = torch.tensor(np.logical_or(terminated, truncated), dtype=torch.float32)

            ob = next_ob.clone()
            if terminated or truncated:
                ob = torch.tensor(env.reset()[0], dtype=torch.float32)
                episodes += 1

        # estimate returns and advantages using GAE
        with torch.no_grad():
            values = agent.critic(obs)
            next_values = agent.critic(next_obs)
            advantages = gae(values, next_values, rewards, dones, agent.discount, agent.gae_param)
            returns = advantages + values.squeeze(1)

        # update policy and value function using mini-batches
        mb_size = agent.minibatch_size
        b_indices = np.arange(b_size)
        for step in range(agent.optim_epochs):
            np.random.shuffle(b_indices)
            for i in range(num_minibatches):
                start, end = mb_size * i, mb_size * (i+1) # inclusive, non-inclusive
                mb_indices = b_indices[start:end]
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_lps = log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                # normalize advantages every mini-batch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_lps = agent.actor.get_log_probs(mb_obs, mb_actions)
                new_values = agent.critic(mb_obs)
                log_ratio = new_lps - mb_lps
                ratio = torch.exp(log_ratio)

                # implement value clipping
                value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()
                agent.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.max_grad_norm)
                agent.critic_optim.step()

                policy_loss = -clipped_loss(ratio, mb_advantages, agent.policy_clip)
                agent.actor_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.max_grad_norm)
                agent.actor_optim.step()

                # for plotting
                total_rewards.append(np.sum(rewards.cpu().numpy()) / episodes)


        if update % 10 == 0:
            plot(global_step, total_rewards)
            # agent.save()




if __name__ == '__main__':
    train('CartPole-v1', 1, continuous=False)
