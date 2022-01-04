import numpy as np

from a2cSingleAgentHarvest import a2c
from social_dilemma.environment.harvest import HarvestEnv
from Hyperparams import *

if __name__ == "__main__":


    # Constants
    GAMMA = 0.99
    num_steps = 100
    max_episodes = 1000
    render_env = False
    hidden_size, learning_rate = get_6_batch()

    repetition = 3

    sums_rewards = []
    for i in range(0, repetition):
        env = HarvestEnv(num_agents=1)
        rewards, sum_rewards = a2c(env, GAMMA=GAMMA, num_steps=num_steps, max_episodes=max_episodes,
                                   render_env=render_env,
                                   learning_rate=learning_rate, hidden_size=hidden_size)
        sums_rewards.append(sum_rewards)
        print("repetition ", i, " sum_rewards: ", sum_rewards)

    print("mean sum all rewards: ", np.mean(sums_rewards))

    '''
    with open('tuning_results/results.txt', 'w') as f:
        hypeparam_str = f"Hyperparameters : \n" \
                        f"Hidden Size = {hidden_size}\n" \
                        f"Learning rate = {learning_rate}\n" \
                        f"Constant : \n" \
                        f"Gamma = {GAMMA} \n" \
                        f"num_steps = {num_steps}\n" \
                        f"max_episodes = {max_episodes} \n\n"
        results_str = f"Results : \n{str(rewards)}"
        filecontent = hypeparam_str + results_str
        f.write(filecontent)
    '''
