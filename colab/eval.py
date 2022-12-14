import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from models import *

# Visualize agents in action

ATARI_ENVS = {
    'PONG': {
        'name': 'PongNoFrameskip-v4',
        'num_actions': 2,
        'action_offset': 4
    }
    # Add more Atari envs
}
ATARI_PATH = './checkpoints/atari_dqn.pth'

def atari(env):
    name = env['name']
    num_actions = env['num_actions']
    action_offset = env['action_offset']

    env = gym.make(name, render_mode='human')
    env = gym.wrappers.AtariPreprocessing(
        env = env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)

    state = env.reset()[0]
    model = AtariCNN(4, num_actions, exploit=True)
    model.load_state_dict(torch.load(ATARI_PATH, map_location=torch.device('cpu')))

    while 1:
        action = model.act(state)

        next_s, r, done, _, _ = env.step(action + action_offset)

        state = next_s
        if done:
            break


if __name__ == '__main__':
    atari(ATARI_ENVS['PONG'])
