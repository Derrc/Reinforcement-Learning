import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from models import Critic, Actor
from utils import estimate_advantages, rollout, flat_grad, kl_divergence, conjugate_gradient, gae
import gymnasium as gym

# Agents for Different Algorithms 
# TODO: add plotting/Adaptive KL Penalty/TRPO continuous action space

# PPO Agent (clipped surrogate function)
# TODO: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
class PPOAgent():
    def __init__(self, env, obs_dim, action_dim, continuous, batch_size=32, discount_factor=0.99, actor_lr=1e-4, critic_lr=1e-4, delta=0.2,
                lamda=0.95, train_epochs=50, optim_steps=10, max_steps=1000, action_low=-1, action_high=1):
        # Set hyperparameters
        self.env = env
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.delta = delta
        self.lamda = lamda
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

        # model paths
        self.actor_path = 'ppo_actor.pth'
        self.critic_path = 'ppo_critic.pth'

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
            with torch.no_grad():
                advantages = [gae(self.critic(state), self.critic(next_state), reward, self.discount_factor, self.lamda) for state, _, _, reward, next_state in rollouts]
                advantages = torch.cat(advantages).flatten()
                # compute target return to fit critic = advantages + values
                target_return = self.critic(states) + advantages
                advantages = (advantages - advantages.mean()) / advantages.std()

            # optimize lower bound
            for _ in range(self.optim_steps):
                # get values of states with critic
                values = self.critic(states).squeeze(1)
                # get log_prob of target actions following current actor
                log_probs = self.actor.get_log_probs(states, target_actions)

                # update critic network
                self.critic_optim.zero_grad()
                critic_loss = (values - target_return).pow(2).mean()
                critic_loss.backward()
                self.critic_optim.step()

                # update actor network
                self.actor_optim.zero_grad()
                # compute clipped surrogate loss
                surrogate_loss = -self.clipped_surrogate_loss(log_probs, target_log_probs, advantages)
                surrogate_loss.backward()
                self.actor_optim.step()

            print(f'Rewards: {np.mean(rollout_rewards):.3f}')
    
    # load models
    def load(self):
        self.actor.load_state_dict(torch.load(self.actor_path))
        self.critic.load_state_dict(torch.load(self.critic_path))

    # save models
    def save(self):
        torch.save(self.actor.state_dict(), self.actor_path)
        torch.save(self.critic.state_dict(), self.critic_path)

    # plot: reward, entropy
    def plot(self):
        pass


# TRPO Agent: currently only supports discrete action spaces
class TRPOAgent():
    def __init__(self, env, obs_dim, action_dim, batch_size=16, discount_factor=0.99, critic_lr=1e-3, delta=0.01, 
                train_epochs=50, max_steps=1000):
        self.env = env
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.delta = delta
        self.train_epochs = train_epochs
        self.max_steps = max_steps
        # initialize actor/critic
        self.actor = Actor(obs_dim, action_dim, continuous=False)
        self.critic = Critic(obs_dim)
        # initialize optimizers
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # surrogate loss -> L(theta): policy gradient of objective function
    # grad(L(theta)) = policy gradient(g)
    def surrogate_loss(self, advantages, new_policy, old_policy):
        return (new_policy / old_policy * advantages).mean()

    # update critic
    def update_critic(self, advantages):
        self.critic_optim.zero_grad()
        loss = advantages.pow(2).mean()
        loss.backward()
        self.critic_optim.step()

    # update actor
    def update_actor(self, max_step):
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            g = max_step[n:n + numel].view(p.shape)
            p.data += g
            n += numel
        
    def train(self):
        self.actor.train()
        for epoch in range(self.train_epochs):
            # rollout with current policy
            rollouts, rollout_rewards = rollout(self.env, self.actor, num_rollouts=self.batch_size, max_steps=self.max_steps)
            states = torch.cat([r.state for r in rollouts])
            actions = torch.cat([r.action for r in rollouts]).flatten()
            old_log_probs = torch.cat([r.log_prob for r in rollouts]).flatten()
            old_policy = torch.exp(old_log_probs)

            # estimate advantages
            advantages = [estimate_advantages(self.critic, state, next_state[-1], reward, self.discount_factor) for state, _, _, reward, next_state in rollouts]
            advantages = torch.cat(advantages, dim=0).flatten()
            # normalize advantages
            advantages = (advantages - advantages.mean()) / advantages.std()

            # update critic
            self.update_critic(advantages)

            parameters = list(self.actor.parameters())

            # compute surrogate loss, taking different distributions for KL Div
            new_probs = self.actor(states)
            dist = Categorical(new_probs)
            new_policy = torch.exp(dist.log_prob(actions))
            loss = self.surrogate_loss(advantages, new_policy, old_policy.detach())
            
            # compute policy gradient for surrogate objective function
            g = flat_grad(loss, parameters, retain_graph=True)

            # compute fisher vector product (Fisher Information Matrix * v)
            def fisher_vector_product(v):
                kl = kl_divergence(new_probs.detach(), new_probs)
                kl_grad = flat_grad(kl, parameters, create_graph=True)

                kl_v = kl_grad.dot(v)
                # FIM = 2nd grad of kl
                kl_v_grad = flat_grad(kl_v, parameters, retain_graph=True).detach()

                return kl_v_grad + 0.1 * v


            # compute conjugate gradient and max step size
            search_dir = conjugate_gradient(fisher_vector_product, g, nsteps=10) # 10 taken from TRPO paper
            step_size = torch.sqrt(2 * self.delta / (torch.dot(search_dir, fisher_vector_product(search_dir))))
            max_step = step_size * search_dir

            # Line search to increase update size but still remain in trust region
            # TODO: implement own line search
            old_policy = new_policy
            old_probs = new_probs
            def criterion(step):
                self.update_actor(step)
                with torch.no_grad():
                    new_probs = self.actor(states)
                    dist = Categorical(new_probs)
                    new_policy = torch.exp(dist.log_prob(actions))

                    new_loss = self.surrogate_loss(advantages, new_policy, old_policy.detach())
                    new_kl = kl_divergence(old_probs, new_probs)
                    
                loss_improve = new_loss - loss

                if loss_improve > 0 and new_kl <= self.delta:
                    return True

                self.update_actor(-step)
                return False

            i = 0
            while not criterion((0.9 ** i) * max_step) and i < 10:
                i += 1


            print(f'{epoch}: Reward: {np.mean(rollout_rewards):.3f}')


# testing
if __name__ == '__main__':
    discrete_env = gym.make('CartPole-v1', render_mode='rgb_array')
    discrete_state_dim = discrete_env.observation_space.shape[0]
    discrete_action_dim = discrete_env.action_space.n
    agent = PPOAgent(discrete_env, discrete_state_dim, discrete_action_dim, train_epochs=1000, continuous=False, max_steps=1000)
    # agent = TRPOAgent(discrete_env, discrete_state_dim, discrete_action_dim)
    agent.train()
    
    cont_env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
    cont_state_dim = cont_env.observation_space.shape[0]
    cont_action_dim = cont_env.action_space.shape[0]
    agent = PPOAgent(cont_env, cont_state_dim, cont_action_dim, continuous=True, max_steps=128)
    # agent.train()


        