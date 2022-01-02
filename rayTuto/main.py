
import ray
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from rayTuto.models.BaselineModel import BaselineModel
from social_dilemma.environment.harvest import HarvestEnv
from ray.tune.registry import register_env
def pretty_print(result):
    print(f"episode_len_mean : {result['episode_len_mean']}")
    print(f"episode_reward_mean : {result['episode_reward_mean']}")

def env_creator(_):
    return  HarvestEnv(num_agents=2)
if __name__ == "__main__":
    ray.init() # Skip or set to ignore if already called
    env = env_creator
    register_env("harvest", env)
    model_name = "baseline"
    ModelCatalog.register_custom_model(model_name, BaselineModel)

    single_env = env_creator(2)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    #obs_space = env.observation_space
    #act_space = env.action_space

    num_agents =  2
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

    config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 2,
            "multiagent" : {"policies" : policy_graphs, "policy_mapping_fn": policy_mapping_fn},
            'model': {
              "custom_model": "baseline",
                "use_lstm": False,
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
                "custom_model_config": {
                    "cell_size": lstm_cell_size,
                    "num_other_agents": num_agents - 1,
                },

            }
          }
    """
    config = {'gamma': 0.9,
              'lr': 1e-2,
              'num_workers': 2,

              }
              """

    trainer = agents.a3c.A2CTrainer(env="harvest", config=config)

    for i in range (5):
        results = trainer.train()
        pretty_print(results)
