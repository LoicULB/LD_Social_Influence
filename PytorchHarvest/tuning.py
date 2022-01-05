import numpy as np
from matplotlib import pyplot as plt

from a2cSingleAgentHarvest import a2c
from social_dilemma.environment.harvest import HarvestEnv
from Hyperparams import *

def plot_curve(scores, figure_file, title):
    nb_episodes = np.shape(scores)[1]
    x_plot = [i + 1 for i in range(nb_episodes)]
    y_plot = np.array(scores)
    y_mean = np.mean(y_plot, axis=0)
    y_std = np.std(y_plot, axis=0)

    plt.plot(x_plot, y_mean, '-', color='gray')
    plt.fill_between(x_plot, y_mean - y_std, y_mean + y_std,
                     color='blue', alpha=0.2)

    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig(figure_file)

if __name__ == "__main__":

    # Constants
    GAMMA = 0.99
    num_steps = 300
    max_episodes = 800 #400
    render_env = False
    hidden_size, learning_rate = get_6_batch()

    repetition = 4

    rewards_repetitions = []
    for i in range(0, repetition):
        env = HarvestEnv(num_agents=1)
        rewards, sum_rewards, all_rewards = a2c(env, GAMMA=GAMMA, num_steps=num_steps, max_episodes=max_episodes,
                                   render_env=render_env,
                                   learning_rate=learning_rate, hidden_size=hidden_size)

        rewards_repetitions.append(all_rewards)
        print("repetition ", i, " sum_rewards: ", sum_rewards)

    title = f"G : {GAMMA} | n steps : {num_steps} | epi : {max_episodes} | nb_rep : {repetition} lr : {learning_rate} | hs : {hidden_size}"
    plot_curve(rewards_repetitions, "tmp/plots/test", title)

    print("mean sum all rewards: ", np.mean(rewards_repetitions))

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


