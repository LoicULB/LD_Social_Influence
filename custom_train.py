
import ray
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from models.baseline_model import BaselineModel
from models.moa_model import MOAModel
from social_dilemmas.envs.harvest import HarvestEnv
from ray.tune.registry import register_env
import matplotlib.pyplot as plt
import numpy as np
from algorithms import a3c_moa
def plot_rewards_evolution(all_rewards, GAMMA, num_steps, max_episodes,learning_rate, hidden_size, nb_agents):
    
    plt.title(
        f"Gamma : {GAMMA} | n steps : {num_steps} \n learning rate : {learning_rate} | hidden size : {hidden_size} |nb agents : {nb_agents}")
    plt.plot(all_rewards)
    #plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    
def plot_curve(scores, GAMMA, num_steps, max_episodes,learning_rate, hidden_size, nb_agents, algo:str):

    nb_episodes = np.shape(scores)[1]
    x_plot = [i + 1 for i in range(nb_episodes)]
    y_plot = np.array(scores)
    y_mean = np.mean(y_plot, axis=0)
    y_std = np.std(y_plot, axis=0)

    plt.plot(x_plot, y_mean, '-', color='gray')
    plt.fill_between(x_plot, y_mean - y_std, y_mean + y_std,
                     color='blue', alpha=0.2)

    plt.title(f"Gamma : {GAMMA} | n steps : {num_steps} \n learning rate : {learning_rate} | hidden size : {hidden_size} |nb agents : {nb_agents}")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()
    #
    #title = f"S_{nb_steps}s_{nb_episodes}e_{learning_rate}lr_{algo}.png"
    #plt.savefig(title)

def pretty_print(result):
    print(f"episode_len_mean : {result['episode_len_mean']}")
    print(f"episode_reward_mean : {result['episode_reward_mean']}")

def env_creator(_):
    #pass
    return  HarvestEnv(num_agents=1)

def defineA2CTrainer(nb_agents: int , nb_steps : int , gamma, lr):
    env = env_creator
    register_env("harvest", env)
    model_name = "baseline"
    ModelCatalog.register_custom_model(model_name, BaselineModel)

    single_env = env_creator(nb_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space


    num_agents =  nb_agents
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": model_name}

    # Create 1 distinct policy per agent
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs["agent-" + str(i)] = gen_policy()


    def policy_mapping_fn(agent_id):
        return agent_id


    conv_filters = [[6, [3, 3], 1]]
    fcnet_hiddens = [32, 32]
    lstm_cell_size = 128

    config = {
       # 'render_env' : True,
       'monitor' : True,
        'horizon' : nb_steps,
        'gamma':gamma,
          'lr': lr,
          'num_workers': 2,
            "multiagent" : {"policies" : policy_graphs, "policy_mapping_fn": policy_mapping_fn},
            'model': {
              "custom_model": "baseline",
                "use_lstm": False,
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
                "custom_options": {
                    "cell_size": lstm_cell_size,
                    "num_other_agents": num_agents - 1,
                },

            }
          }

  

    return agents.a3c.A2CTrainer(env="harvest", config=config)

def define_moa_trainer(nb_agents: int , nb_steps : int , gamma, lr):
    
    env = env_creator
    register_env("harvest", env)
    model_name = "moa"
    ModelCatalog.register_custom_model(model_name, MOAModel(obs_space, action_space, num_outputs, model_config, name))

    single_env = env_creator(nb_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space


    num_agents =  nb_agents
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": model_name}

    # Create 1 distinct policy per agent
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs["agent-" + str(i)] = gen_policy()


    def policy_mapping_fn(agent_id):
        return agent_id


    conv_filters = [[6, [3, 3], 1]]
    fcnet_hiddens = [32, 32]
    lstm_cell_size = 128

    config = {
       # 'render_env' : True,
       #'env_name' : "harvest",

        'horizon' : nb_steps,
        'gamma':gamma,
          'lr': lr,
          'num_workers': 2,
            "multiagent" : {"policies" : policy_graphs, "policy_mapping_fn": policy_mapping_fn},
            'model': {
              "custom_model": "baseline",
                "use_lstm": False,
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
                "custom_options": {
                    "cell_size": lstm_cell_size,
                    "num_other_agents": num_agents - 1,
                },

            }
          }

  

    return agents.a3c.A3CTrainer(env="harvest", config=config)
    #return a3c_moa.build_a3c_moa_trainer(config)
def get_first_batch():
    gamma = 0.99
    lr = 0.01
    nb_steps = 50
    nb_epi = 250
    nb_agents = 1
    return gamma, lr, nb_steps, nb_epi, nb_agents
def get_speedy_batch():
    gamma = 0.99
    lr = 0.01
    nb_steps = 5
    nb_epi = 5
    nb_agents = 1
    return gamma, lr, nb_steps, nb_epi, nb_agents
def get_first_exp():
    gamma = 0.99
    lr = 0.001
    nb_steps = 100
    nb_epi = 300
    nb_agents = 1
    return gamma, lr, nb_steps, nb_epi, nb_agents
    
if __name__ == "__main__":
    ray.init() # Skip or set to ignore if already called
    gamma, lr, nb_steps , nb_epi,  nb_agents = get_speedy_batch()
    trainer = defineA2CTrainer(nb_agents, nb_steps, gamma, lr)
    #trainer = define_moa_trainer(nb_agents, nb_steps, gamma, lr)
    rewards_plot = []
    nb_repetitions = 1
    for rep in range(nb_repetitions):
        rep_results = []

        for i in range(nb_epi):
            results = trainer.train()
            # pretty_print(results)
            print(f"Episode : {i}")
            rep_results.append(results['episode_reward_mean'])
        rewards_plot.append(rep_results)

    plot_curve(rewards_plot, gamma, nb_steps, nb_epi, lr, 32, nb_agents, "A2CRllib")
