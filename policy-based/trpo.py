from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# Trust-Region Policy Optimization
# (Using Actor-Critic Network to estimate Advantages)

# Number of episodes
EPISODES = 1000
# Max steps per episode
MAX_STEPS = 1000
# Discount factor
GAMMA = 0.99
# Delta
DELTA = 0.0001 # constraint value used from TRPO paper
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
Rollout = namedtuple('Rollout', ['state', 'action', 'next_state', 'action_prob', 'advantage'])

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
        x = torch.from_numpy(x)
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
        x = torch.from_numpy(x)
        x = self.layers(x)
        actions = F.softmax(x, dim=1)
        return actions

    # get action sampled from stochastic policy
    def get_action(self,a):
        return np.random.choice(self.action_dim,p=a.detach().numpy())

    # assuming not batched input state
    def act(self, state):
        state = np.expand_dims(state, 0)
        actions = self.forward(state).squeeze(0)
        action = self.get_action(actions)
        action_prob = actions[action]
        return action, action_prob


# rollout -> run trajectories gain experience from current policy
# TODO: implement GAE to estimate advantages, current implementation is naive and clunky
def rollout(actor, critic, steps=1000):
    rollouts = []
    rewards, episode_rewards = [], []
    state = env.reset()[0]
    I = 1
    for _ in range(steps):
        action, action_prob = actor.act(state)

        next_s, r, done, _, _ = env.step(action)

        advantage = r + (1-done) * GAMMA * critic(next_s) - critic(state)
        advantage *= I
        
        rollouts.append(Rollout(state, action, next_s, action_prob, advantage))
        rewards.append(r)

        state = next_s
        I *= GAMMA
        if done:
            episode_rewards.append(np.sum(rewards))
            rewards = []
            state = env.reset()[0]
            I = 1

    return rollouts, episode_rewards
        

# surrogate loss -> L(theta): policy gradient of objective function
# grad(L(theta)) = policy gradient(g)
def surrogate_loss(advantages, new_policy, old_policy):
    advantages.detach()
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



# compute KL-Divergence
def kl_divergence(new_policy, old_policy):
    new_policy.detach()
    return (new_policy * (new_policy.log() - old_policy.log())).sum()


# compute Fisher-Information Matrix vector product for use in conujugate gradient method
# F = 2nd grad of KL
def fisher_vector_product(d_kl, b, parameters):
    b.detach()
    fisher = flat_grad(d_kl @ b, parameters, retain_graph=True)
    return fisher


# conjugate gradient method
def conjugate_gradient(d_kl, b, parameters, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = fisher_vector_product(d_kl, p, parameters)
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

    for _ in range(iterations):
        # rollout with current policy
        rollouts, rewards = rollout(actor, critic, steps=500)
        states = np.array([r.state for r in rollouts])
        actions = torch.tensor([r.action for r in rollouts])
        old_policy = torch.stack([r.action_prob for r in rollouts])
        advantages = torch.stack([r.advantage for r in rollouts])

        # normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        # update critic
        update_critic(critc_optim, advantages)

        parameters = list(actor.parameters())

        # compute surrogate loss
        new_probs = actor(states)
        new_policy = torch.gather(new_probs, 1, actions.unsqueeze(1)).squeeze(1)
        loss = surrogate_loss(advantages, new_policy, new_policy.detach())
        # compute policy gradient for surrogate objective function
        g = flat_grad(loss, parameters, retain_graph=True)

        # compute kl divergence and grad of kl
        kl = kl_divergence(new_policy, old_policy)
        d_kl = flat_grad(kl, parameters, create_graph=True)

        # compute conjugate gradient and max step size
        search_dir = conjugate_gradient(d_kl, g, parameters, nsteps=10) # 10 taken from TRPO paper
        step_size = torch.sqrt(2 * DELTA / (search_dir @ fisher_vector_product(d_kl, search_dir, parameters)))
        max_step = step_size * search_dir
        # update_actor(actor, max_step)

        # Line search to increase update size but still remain in trust region

        print(f'Reward: {np.mean(rewards)}')

        


if __name__ == '__main__':
    train(5)
    
