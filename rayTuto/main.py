
import ray
from ray.rllib import agents
from social_dilemma.environment.harvest import HarvestEnv
from ray.tune.registry import register_env
def pretty_print(result):
    print(f"episode_len_mean : {result['episode_len_mean']}")
    print(f"episode_reward_mean : {result['episode_reward_mean']}")

def env_creator(_):
    return  HarvestEnv(num_agents=2)
if __name__ == "__main__":
    ray.init() # Skip or set to ignore if already called
    config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 2,
          'train_batch_size': 1000,
          'model': {
              'fcnet_hiddens': [128, 128]
          }}
    env = env_creator
    register_env("harvest", env)
    trainer = agents.a3c.A2CTrainer(env="harvest", config=config)

    for i in range (10):
        results = trainer.train()
        pretty_print(results)
