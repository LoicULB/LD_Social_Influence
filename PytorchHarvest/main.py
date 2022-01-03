import sys
import torch

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

from PytorchHarvest.Hyperparams import *
from a2cSingleAgentHarvest import a2c
from social_dilemma.environment.harvest import HarvestEnv

if __name__ == "__main__":

    # Constants
    GAMMA = 0.99
    num_steps = 300
    max_episodes = 30
    render_env = False
    hidden_size, learning_rate = get_8_batch()

    env = HarvestEnv(num_agents=1)
    _, sum_rewards = a2c(env, GAMMA=GAMMA, num_steps=num_steps, max_episodes=max_episodes, render_env=render_env,
                         learning_rate=learning_rate, hidden_size=hidden_size)

