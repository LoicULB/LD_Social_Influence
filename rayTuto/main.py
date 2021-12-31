
import ray
from ray.rllib import agents

if __name__ == "__main__":
    ray.init() # Skip or set to ignore if already called
    config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 2,
          'train_batch_size': 1000,
          'model': {
              'fcnet_hiddens': [128, 128]
          }}
    trainer = agents.a3c.A2CTrainer(env='CartPole-v0', config=config)
    results = trainer.train()
    print("Choco")