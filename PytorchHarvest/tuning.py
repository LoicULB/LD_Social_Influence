

from a2cSingleAgentHarvest import a2c
from social_dilemma.environment.harvest import HarvestEnv
from Hyperparams import get_first_batch
if __name__ == "__main__":
    hidden_size, learning_rate = get_first_batch()
    # Constants
    GAMMA = 0.99
    num_steps = 300
    max_episodes = 50
    env = HarvestEnv(num_agents=1)
    rewards,_ = a2c(env)
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
