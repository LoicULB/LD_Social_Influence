
from harvest import HarvestEnv


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = HarvestEnv()
    print(env.action_space)
    env.setup_agents()
    print()
    env.render()

