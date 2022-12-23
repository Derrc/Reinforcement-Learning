import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from models import Critic, Actor
from utils import estimate_advantages, rollout, flat_grad, kl_divergence, conjugate_gradient
import gymnasium as gym

# Trust Region Policy Optimization (TRPO)
# TODO: refactor and recode (messy)

class TRPOAgent():
    def __init__(self, env_name, **hyperparameters):
        # initialize environment
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        # currently onl for discrete envs
        self.action_dim = self.env.action_space.n

        self.init_hyperparameters(hyperparameters)

        # initialize actor/critic
        self.actor = Actor(self.obs_dim, self.action_dim, continuous=False)
        self.critic = Critic(self.obs_dim)
        # initialize optimizers
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    # initialize hyperparameters
    def init_hyperparameters(self, hyperparameters):
        self.train_epochs = 100
        self.batch_size = 32
        self.discount_factor = 0.99
        self.delta = 0.01
        self.max_steps = 500
        self.critic_lr = 1e-3
        self.seed = None

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
			# set seed
            torch.manual_seed(self.seed)

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
    agent = TRPOAgent('CartPole-v1')
    agent.train()



        