import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# TD(0) Advantage Actor Critic 

# Number of episodes to train over
EPISODES = 1000
# Discount factor
GAMMA = 0.99
# Learning rate
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
# Max steps per episode
MAX_STEPS = 1000
# Solved if average over 195
SOLVED_SCORE = 195

# model paths
PATH_ACTOR = './models/td0_actor.pth'
PATH_CRITIC = './models/td0_critic.pth'

# ENV
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


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
        # input -> 75 x 80 frame
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    # outputs action, log prob of action
    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.layers(x)
        # softmax to get prob of actions
        actions = F.softmax(x, dim=0)
        # get action sampled from stochastic policy
        action = self.get_action(actions)
        # get log_prob of sampled action
        log_prob_action = torch.log(actions.squeeze(0))[action]
        # at, log pi(at|st)
        return action, log_prob_action

    # get action sampled from stochastic policy
    def get_action(self,a):
        return np.random.choice(2,p=a.detach().numpy())


def train():
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optim = torch.optim.SGD(actor.parameters(), lr=ACTOR_LR)
    critic_optim = torch.optim.SGD(critic.parameters(), lr=CRITIC_LR)

    actor.train()
    critic.train()
    total_rewards = []
    for episode in range(EPISODES):
        # get initial state
        state = env.reset()[0]
        # decay per timestep (because J(theta) = V(s0))
        I = 1
        rewards = []
        for _ in range(MAX_STEPS):
            # get sampled action, log_prob of action from actor
            action, log_prob = actor(state)

            # take step in env
            next_s, r, done, _, _ = env.step(action)

            # calculate advantage (TD error)
            advantage = r + (1-done)*GAMMA*critic(next_s) - critic(state)
            
            # update parameters
            critic_loss = pow(advantage, 2) * I
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            actor_loss = -(log_prob * advantage.detach() * I)
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            state = next_s
            rewards.append(r)
            I *= GAMMA

            if done:
                break
        
        total_rewards.append(np.sum(rewards))
        if episode % 100 == 0:
            print(f'EPISODE: {episode}, MEAN: {np.mean(total_rewards[-100:])}')
        if np.mean(total_rewards[-100:]) > SOLVED_SCORE:
                print(f'Game is Solved!')
                break

    # save models
    torch.save(actor.state_dict(), PATH_ACTOR)
    torch.save(critic.state_dict(), PATH_CRITIC)

    # plot total rewards
    plt.plot(total_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.title('Reward of Actor-Critic TD(0) on CartPole')

    reg = LinearRegression().fit(np.arange(len(total_rewards)).reshape(-1, 1), np.array(total_rewards).reshape(-1, 1))
    y_pred = reg.predict(np.arange(len(total_rewards)).reshape(-1, 1))
    plt.plot(y_pred)
    plt.show()


def eval():
    actor = Actor(state_dim, action_dim)
    actor.eval()

    # load model
    if os.path.exists(PATH_ACTOR):
        actor.load_state_dict(torch.load(PATH_ACTOR))

    eval_episodes = 30
    eval_steps = 10000
    total_rewards = []
    for episode in range(eval_episodes):
        state = env.reset()[0]
        rewards = []
        for _ in range(eval_steps):
            action = actor(state)[0]
        
            next_s, r, done, _, _ = env.step(action)

            state = next_s
            rewards.append(r)

            if done:
                break
        
        sum = np.sum(rewards)
        total_rewards.append(sum)
        print(f'EPISODE: {episode+1}, REWARD: {sum}')

    print(f'MEAN REWARD: {np.mean(total_rewards)}')


if __name__ == '__main__':
    # train()
    eval()