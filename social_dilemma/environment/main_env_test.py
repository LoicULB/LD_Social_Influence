
from social_dilemma.environment.harvest import HarvestEnv


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = HarvestEnv(num_agents=1)
    env.reset()

    env.setup_agents()


    print("orientation agent 0: ",env.agents["agent-0"].orientation) # TEST


    # render the current environment
    env.render("current-environment at time 0")

    dic = {"agent-0": 3}


    env.step(dic)
    # render the current environment
    env.render("current-environment at time 1")

