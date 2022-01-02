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
from a2c import  a2c
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    a2c(env)