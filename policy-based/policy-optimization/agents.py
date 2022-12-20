import torch
import torch.nn as nn
import numpy as np
from models import Critic, Actor
from utils import estimate_advantages, get_cumulative_rewards, rollout
import gymnasium as gym

# Agents for Different Algorithms
# TODO: add TRPO/add plotting/Adaptive KL Penalty
class PPOAgent():
    def __init__(self, env, obs_dim, action_dim, batch_size, discount_factor, actor_lr, critic_lr, delta, 
                train_epochs, optim_steps, max_steps, action_low=-1, action_high=1, continuous=False):
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.delta = delta
        self.train_epochs = train_epochs
        self.optim_steps = optim_steps
        self.max_steps = max_steps
        self.continuous = continuous
        # initialize Actor/Critic
        if self.continuous:
            self.actor = Actor(obs_dim, action_dim, action_low=action_low, action_high=action_high, continuous=True)
        else:
            self.actor = Actor(obs_dim, action_dim, continuous=False)
        self.critic = Critic(obs_dim)

        # initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def clipped_surrogate_loss(self, new_policy, old_policy, advantages):
        # compute ratio from log probs of actions
        ratio = torch.exp(new_policy - old_policy)
        clipped_ratio = torch.clamp(ratio, 1 - self.delta, 1 + self.delta).detach()
        return torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    def train(self):
        self.actor.train()
        self.critic.train()
        for _ in range(self.train_epochs):
            # rollouts
            rollouts, rollout_rewards = rollout(self.env, self.actor, num_rollouts=self.batch_size, max_steps=self.max_steps)
            states = torch.cat([r.state for r in rollouts])
            target_actions = torch.cat([r.action for r in rollouts]).flatten()
            # log_prob of actions following target actor
            target_log_probs = torch.cat([r.log_prob for r in rollouts]).flatten()

            # compute advantages and normalize
            advantages = [estimate_advantages(self.critic, state, next_state[-1], reward, self.discount_factor) for state, _, _, reward, next_state in rollouts]
            advantages = torch.cat(advantages).flatten().detach()
            advantages = (advantages - advantages.mean()) / advantages.std()

            # compute cumulative rewards
            cumulative_rewards = torch.cat([get_cumulative_rewards(r.reward, self.discount_factor) for r in rollouts])

            for _ in range(self.optim_steps):
                # get values of states with critic
                values = self.critic(states).squeeze(1)
                # get log_prob of target actions following current actor
                log_probs = self.actor.get_log_probs(states, target_actions)

                # update critic network
                self.critic_optim.zero_grad()
                critic_loss = (values - cumulative_rewards).pow(2).mean()
                critic_loss.backward()
                self.critic_optim.step()

                # update actor network
                self.actor_optim.zero_grad()
                # compute clipped surrogate loss
                surrogate_loss = -self.clipped_surrogate_loss(log_probs, target_log_probs, advantages)
                surrogate_loss.backward()
                self.actor_optim.step()

            print(f'Rewards: {np.mean(rollout_rewards):.3f}')



if __name__ == '__main__':
    discrete_env = gym.make('CartPole-v1', render_mode='rgb_array')
    discrete_state_dim = discrete_env.observation_space.shape[0]
    discrete_action_dim = discrete_env.action_space.n
    # agent = PPOAgent(discrete_env, discrete_state_dim, discrete_action_dim, batch_size=16, discount_factor=0.99, actor_lr=1e-3, critic_lr=1e-3, delta=0.2, train_epochs=50, optim_steps=10, max_steps=1000, continuous=False)
    
    cont_env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
    cont_state_dim = cont_env.observation_space.shape[0]
    cont_action_dim = cont_env.action_space.shape[0]
    agent = PPOAgent(cont_env, cont_state_dim, cont_action_dim, batch_size=16, discount_factor=0.99, actor_lr=1e-3, critic_lr=1e-3, delta=0.2, train_epochs=50, optim_steps=10, max_steps=128, continuous=True)
    agent.train()

        