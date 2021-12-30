import gym
from NicoA2C.Agent import A2CAgent

if __name__ == "__main__":
    env_name = "CartPole-v0"
    agent = A2CAgent(env_name)
    agent.run()
