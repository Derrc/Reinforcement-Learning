from collections import namedtuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym

# Proximal Policy Optimization for Discrete Action Spaces
# TODO: Refactor discrete/continuous code together, implement variations of PPO as seen in paper
# - parameter sharing between policy and value network
# - changing loss function to include value loss and entropy regularization


# Number of rollouts per iteration
ROLLOUTS = 10
# Max steps per episode
MAX_STEPS = 1000
# Discount factor
GAMMA = 0.99
# Limiting constant on policy ratio (1 - DELTA, 1 + DELTA)
DELTA = 0.2
# iterations for optimizing surrogate function
K = 10
# Learning rate
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
# Solved if average over 195
SOLVED_SCORE = 195
# model paths
PATH_ACTOR = './models/ppo_actor.pth'
PATH_CRITIC = './models/ppo_critic.pth'

# env
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# rollout tuple
Rollout = namedtuple('Rollout', ['state', 'action', 'log_prob', 'reward', 'next_state'])

# Value Network: V(s)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# Policy Network: pi(a|s)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),                
            nn.Linear(128, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, x):
        x = self.layers(x)
        x = torch.softmax(x, dim=1)
        return x

    def get_log_probs(self, state, actions):
        curr_actions = self.forward(state)
        dist = Categorical(curr_actions)
        log_prob = dist.log_prob(actions)
        return log_prob

    def get_action(self,state):
        state = torch.from_numpy(np.expand_dims(state, 0))
        actions = self.forward(state).flatten()
        action = np.random.choice(self.action_dim,p=actions.detach().numpy())
        log_prob_action = torch.log(actions.squeeze(0))[action]

        return action, log_prob_action

# rollout -> run trajectories gain experience from current policy
# TODO: implement GAE to estimate advantages
def rollout(actor, num_rollouts=ROLLOUTS, max_steps=MAX_STEPS):
    rollouts = []
    rollout_rewards = []
    for _ in range(num_rollouts):
        samples = []
        state = env.reset()[0]
        for _ in range(max_steps):
            action, log_prob = actor.get_action(state)

            next_s, r, done, _, _ = env.step(action)

            samples.append((state, action, log_prob, r, next_s))
            state = next_s

            if done:
                break
        
        # append rollout
        states, actions, log_probs, rewards, next_states = zip(*samples)
        states = torch.stack([torch.from_numpy(state) for state in states])
        next_states = torch.stack([torch.from_numpy(state) for state in next_states])
        actions = torch.tensor(actions).unsqueeze(1)
        log_probs = torch.tensor(log_probs).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)

        rollouts.append(Rollout(states, actions, log_probs, rewards, next_states))
        rollout_rewards.append(torch.sum(rewards))

    return rollouts, rollout_rewards

 
# estimate advantages over each episode
def estimate_advantages(critic, states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)

    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + GAMMA * last_value

    return next_values - values

# compute rewards-to-go
def get_cumulative_rewards(rewards):
    cr = [rewards[-1]]
    for i in range(len(rewards)-2, -1, -1):
        cr.append(rewards[i] + GAMMA * cr[-1])
    cr.reverse()
    return torch.tensor(cr)


# calculate clipped surrogate objective function
def compute_surrogate_loss(new_policy, old_policy, advantages):
    # compute ratio from log probs of actions
    ratio = torch.exp(new_policy - old_policy)
    clipped_ratio = torch.clamp(ratio, 1 - DELTA, 1 + DELTA).detach()
    return torch.min(ratio * advantages, clipped_ratio * advantages).mean()


def train(iterations):
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    actor.train()
    critic.train()
    for iter in range(iterations):
        # rollouts
        rollouts, rollout_rewards = rollout(actor, num_rollouts=ROLLOUTS, max_steps=MAX_STEPS)
        states = torch.cat([r.state for r in rollouts])
        target_actions = torch.cat([r.action for r in rollouts]).flatten()
        # log_prob of actions following target actor
        target_log_probs = torch.cat([r.log_prob for r in rollouts]).flatten()

        # compute advantages and normalize
        advantages = [estimate_advantages(critic, state, next_state[-1], reward) for state, _, _, reward, next_state in rollouts]
        advantages = torch.cat(advantages).flatten().detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # compute cumulative rewards
        cumulative_rewards = torch.cat([get_cumulative_rewards(r.reward) for r in rollouts])

        for _ in range(K):
            # get values of states with critic
            values = critic(states).squeeze(1)
            # get log_prob of target actions following current actor
            log_probs = actor.get_log_probs(states, target_actions)

            # update critic network
            critic_optim.zero_grad()
            critic_loss = (values - cumulative_rewards).pow(2).mean()
            critic_loss.backward()
            critic_optim.step()

            # update actor network
            actor_optim.zero_grad()
            # compute clipped surrogate loss
            surrogate_loss = -compute_surrogate_loss(log_probs, target_log_probs, advantages)
            surrogate_loss.backward()
            actor_optim.step()

        # update target actor weights to current actor
        print(f'Rewards: {np.mean(rollout_rewards):.3f}')

    

if __name__ == '__main__':
    train(100)
