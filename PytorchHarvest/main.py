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

from a2cSingleAgentHarvest import a2c
from social_dilemma.environment.harvest import HarvestEnv

if __name__ == "__main__":
    env = HarvestEnv(num_agents=1)
    a2c(env)
