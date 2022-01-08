
import ray
from ray.rllib import agents



def pretty_print(result):
    print(f"episode_len_mean : {result['episode_len_mean']}")
    print(f"episode_reward_mean : {result['episode_reward_mean']}")


if __name__ == "__main__":
    ray.init() # Skip or set to ignore if already called
    repetitions = 100

    config = {'gamma': 0.9,
              'lr': 1e-2,
              'num_workers': 2, # number of CPU on your PC
              'render_env' : True, #show a video a the environment while training
    }

    trainer = agents.a3c.A2CTrainer(env="CartPole-v0", config=config)

    for i in range (repetitions):
        results = trainer.train()
        pretty_print(results)
