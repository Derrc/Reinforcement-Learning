from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

Rollout = namedtuple('Rollout', ['state', 'action', 'log_prob', 'reward', 'next_state'])

# rollout -> run trajectories gain experience from current policy
def rollout(env, actor, num_rollouts, max_steps):
    rollouts = []
    rollout_rewards = []
    for _ in range(num_rollouts):
        samples = []
        state = env.reset(seed=11)[0]
        for _ in range(max_steps):
            action, log_prob = actor.get_action(torch.from_numpy(state))

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
def estimate_advantages(critic, states, last_state, rewards, gamma):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)

    for i in reversed(range(len(rewards))):
        last_value = next_values[i] = rewards[i] + gamma * last_value

    return next_values - values


def gae(values, next_values, rewards, dones, discount, gae_param):
    advantages = torch.zeros(len(rewards) + 1)

    for i in reversed(range(len(rewards)-1)):
        nonterminal = 1 - dones[i]
        delta = rewards[i] + discount * next_values[i] - values[i]
        advantages[i] = delta + (nonterminal * discount * gae_param * advantages[i+1])

    advantages = advantages[:len(rewards)]
    return advantages


# compute rewards-to-go
def get_cumulative_rewards(rewards, gamma):
    cr = [rewards[-1]]
    for i in reversed(range(1, len(rewards))):
        cr.append(rewards[i] + gamma * cr[-1])
    cr.reverse()
    return torch.tensor(cr)


# compute gradient and flatten
def flat_grad(x, parameters, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    
    grads = torch.autograd.grad(x, parameters, retain_graph=retain_graph, create_graph=create_graph)
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    return torch.cat(grad_flatten)


# compute KL-Divergence between two probability distributions (arrays)
def kl_divergence(old_dist, new_dist):
    return (old_dist * (old_dist.log() - new_dist.log())).mean()


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