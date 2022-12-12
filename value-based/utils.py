import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


# run a random policy through environment to collect states to calculate average Q-Values
def get_evaluation_states(env, episodes):
    evaluation_states = []
    for _ in range(episodes):
        state = env.reset()[0]
        for _ in range(1000):
            evaluation_states.append(state)
            action = env.action_space.sample()

            next_s, r, done, _, _ = env.step(action)

            state = next_s

            if done:
                break
    
    return evaluation_states


# compute the average Q-Value of states as an evaluation metric
def get_average_q_value(model, states):
    with torch.no_grad():
        q_values = model(np.array(states))
    return torch.mean(q_values)


# plot mean reward as well as mean q-values over episodes
def plot_results(total_rewards, total_q_values, model):
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'{model} on CartPole')
    plt.xlim(right=2000)
    plt.ylim(top=500)
    reg = LinearRegression().fit(
        np.reshape(np.arange(len(total_rewards)), (-1, 1)),
        np.reshape(total_rewards, (-1, 1))
    )
    plt.plot(reg.predict(np.reshape(np.arange(len(total_rewards)), (-1, 1))))
    plt.show()

    plt.plot(total_q_values)
    plt.xlabel('Episodes')
    plt.ylabel('Q-Value')
    plt.title(f'{model} Average Q-Value on CartPole')
    plt.show()


def run_trials(trials, train, title):
    rewards = []
    q_values = []
    solved_episodes = []
    for i in range(trials):
        total_rewards, total_q_values, solved_episode = train()
        if solved_episode:
            solved_episodes.append(solved_episode)
        rewards.append(np.mean(total_rewards[-100]))
        q_values.append(np.mean(total_q_values))

        if i == trials-1:
            plot_results(total_rewards, total_q_values, title)
    
    print(f'Mean Reward over Trials: {np.mean(rewards)}')
    print(f'Mean Q-Value over Trials: {np.mean(q_values)}')
    if solved_episodes:
        print(f'Mean Solved Episode over Trials: {np.mean(solved_episodes)}')


# evaluate model in environment
def eval(env, model, PATH, eval_episodes, eval_steps):
    model.eval()
    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))

    eval_episodes = 30
    eval_steps = 10000
    total_rewards = []
    for episode in range(eval_episodes):
        state = env.reset()[0]
        rewards = []
        for _ in range(eval_steps):
            actions = model(state)
            action = np.argmax(actions.detach().numpy())

            next_s, r, done, _, _ = env.step(action)

            state = next_s
            rewards.append(r)
            if done:
                break
        
        total_rewards.append(np.sum(rewards))
        print(f'EPISODE: {episode}, REWARD: {np.sum(rewards)}')

    print(f'MEAN: {np.mean(total_rewards)}')