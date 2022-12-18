from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# Trust-Region Policy Optimization (Currently for discrete action spaces)
# (Using Actor-Critic Network to estimate Advantages)

# Number of rollouts per iteration
ROLLOUTS = 10
# Max steps per episode
MAX_STEPS = 1000
# Discount factor
GAMMA = 0.99
# Delta
DELTA = 0.01 # constraint value used from TRPO paper
# Learning rate
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
# Solved if average over 195
SOLVED_SCORE = 195
# model paths
PATH_ACTOR = './models/td0_actor.pth'
PATH_CRITIC = './models/td0_critic.pth'

# env
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# rollout tuple
Rollout = namedtuple('Rollout', ['state', 'action', 'reward', 'next_state', 'action_prob'])

# Critic Network: S-> Value(S)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            # output V(s)
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(0)


# Actor Network: S -> pi(at|st)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.action_dim = action_dim
    
    # outputs action, log prob of action
    def forward(self, x):
        x = self.layers(x)
        actions = F.softmax(x, dim=1)
        return actions

    # get action sampled from stochastic policy
    def get_action(self,a):
        return np.random.choice(self.action_dim,p=a.detach().numpy())

    # assuming not batched input state
    def act(self, state):
        state = torch.from_numpy(np.expand_dims(state, 0))
        actions = self.forward(state).squeeze(0)
        action = self.get_action(actions)
        action_prob = actions[action]
        return action, action_prob


# rollout -> run trajectories gain experience from current policy
# TODO: implement GAE to estimate advantages, make rollout for continuous state space
def rollout(actor, num_rollouts=ROLLOUTS, max_steps=MAX_STEPS):
    rollouts = []
    rollout_rewards = []
    for _ in range(num_rollouts):
        samples = []
        state = env.reset()[0]
        for _ in range(max_steps):
            action, action_prob = actor.act(state)

            next_s, r, done, _, _ = env.step(action)

            samples.append((state, action, r, next_s, action_prob))
            state = next_s

            if done:
                break
        
        # append rollout
        states, actions, rewards, next_states, action_probs = zip(*samples)
        states = torch.stack([torch.from_numpy(state) for state in states])
        next_states = torch.stack([torch.from_numpy(state) for state in next_states])
        actions = torch.tensor(actions).unsqueeze(1)
        action_probs = torch.tensor(action_probs).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)

        rollouts.append(Rollout(states, actions, rewards, next_states, action_probs))
        rollout_rewards.append(torch.sum(rewards))

    return rollouts, rollout_rewards
        

# surrogate loss -> L(theta): policy gradient of objective function
# grad(L(theta)) = policy gradient(g)
def surrogate_loss(advantages, new_policy, old_policy):
    return (new_policy / old_policy * advantages).mean()



# compute gradient and flatten
def flat_grad(x, parameters, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    
    grads = torch.autograd.grad(x, parameters, retain_graph=retain_graph, create_graph=create_graph)
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    return torch.cat(grad_flatten)

# estimate advantages over each episode
def estimate_advantages(critic, states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)

    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + GAMMA * last_value

    advantages = next_values - values
    return advantages


# compute KL-Divergence
def kl_divergence(old_dist, new_dist):
    old_dist = old_dist.detach()
    return (old_dist * (old_dist.log() - new_dist.log())).sum(-1).mean()

# conjugate gradient method
def conjugate_gradient(Fvp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = Fvp(p)
        # step size in current direction
        alpha = rdotr / torch.dot(p, _Avp)
        # next point
        x += alpha * p
        # remaining error from optimal point
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        # new direction
        p = r + beta * p
        rdotr = new_rdotr
        # if error is low -> close to optimal point -> break
        if rdotr < residual_tol:
            break
    return x


# update critic
def update_critic(optim, advantages):
    optim.zero_grad()
    loss = advantages.pow(2).mean()
    loss.backward()
    optim.step()



# update actor
def update_actor(actor, max_step):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = max_step[n:n + numel].view(p.shape)
        p.data += g
        n += numel



def train(iterations):
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    critc_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    for iter in range(iterations):
        # rollout with current policy
        rollouts, rollout_rewards = rollout(actor, num_rollouts=ROLLOUTS, max_steps=MAX_STEPS)
        states = torch.cat([r.state for r in rollouts])
        actions = torch.cat([r.action for r in rollouts]).flatten()
        old_policy = torch.cat([r.action_prob for r in rollouts]).flatten()

        # estimate advantages
        advantages = [estimate_advantages(critic, state, next_state[-1], reward) for state, _, reward, next_state, _ in rollouts]
        advantages = torch.cat(advantages, dim=0).flatten()
        # normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        # update critic
        update_critic(critc_optim, advantages)

        parameters = list(actor.parameters())

        # compute surrogate loss, taking different distributions for KL Div
        new_probs = actor(states)
        new_policy = torch.gather(new_probs, 1, actions.unsqueeze(1)).squeeze(1)
        loss = surrogate_loss(advantages, new_policy, old_policy.detach())
        
        # compute policy gradient for surrogate objective function
        g = flat_grad(loss, parameters, retain_graph=True)

        def fisher_vector_product(v):
            kl = kl_divergence(new_probs, new_probs)
            kl_grad = flat_grad(kl, parameters, create_graph=True)

            kl_v = kl_grad.dot(v)
            kl_v_grad = flat_grad(kl_v, parameters, retain_graph=True).detach()

            return kl_v_grad + 0.1 * v



        # compute conjugate gradient and max step size
        search_dir = conjugate_gradient(fisher_vector_product, g, nsteps=10) # 10 taken from TRPO paper
        step_size = torch.sqrt(2 * DELTA / (search_dir @ fisher_vector_product(search_dir)))
        max_step = step_size * search_dir

        # Line search to increase update size but still remain in trust region
        # TODO: implement own line search
        old_policy = new_policy
        old_probs = new_probs
        def criterion(step):
            update_actor(actor, step)
            with torch.no_grad():
                new_probs = actor(states)
                new_policy = torch.gather(new_probs, 1, actions.unsqueeze(1)).squeeze(1)

                new_loss = surrogate_loss(advantages, new_policy, old_policy.detach())
                new_kl = kl_divergence(old_probs, new_probs)
                
            loss_improve = new_loss - loss

            if loss_improve > 0 and new_kl <= DELTA:
                return True

            update_actor(actor, -step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1


        print(f'{iter}: Reward: {np.mean(rollout_rewards):.3f}')
        

if __name__ == '__main__':
    train(50)
    
