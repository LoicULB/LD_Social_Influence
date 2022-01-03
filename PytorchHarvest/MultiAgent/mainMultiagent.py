import numpy as np

from a2c_multiagent import a2c
from social_dilemma.environment.harvest import HarvestEnv
from PytorchHarvest.Hyperparams import *

if __name__ == "__main__":


    # Constants
    GAMMA = 0.99
    num_steps = 100
    max_episodes = 31
    render_env = True
    hidden_size, learning_rate = get_7_batch()
    num_agents = 2
    repetition = 10

    sums_rewards = []

    env = HarvestEnv(num_agents=num_agents)
    print("Before run")
    a2c(env, nb_agents=num_agents, GAMMA=GAMMA, num_steps=num_steps, max_episodes=max_episodes,
                               render_env=render_env,
                               learning_rate=learning_rate, hidden_size=hidden_size)

    print("After run")

    #print("mean sum all rewards: ", np.mean(sums_rewards))
