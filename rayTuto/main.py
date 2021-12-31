
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

    obs_space = env.observation_space
    act_space = env.action_space

    """
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": model_name}
    """
    config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 2,
            'model': {
              "custom_model": "baseline",
              # Extra kwargs to be passed to your model's c'tor.
              "custom_model_config": {},
            }
          }

    trainer = agents.a3c.A2CTrainer(env="harvest", config=config)

    for i in range (1):
        results = trainer.train()
        pretty_print(results)
