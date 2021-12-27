import gym
import numpy as np
from ActorCriticAgent import Agent
from harvest import HarvestEnv
from utils import plot_learning_curve
from gym import wrappers

def make_all_agent_play(observations, neuronal_agents ): # : Agent
    actions = {}

    for i in range(len(observations)):
        agent = neuronal_agents[i]                          # --> agent i
        observation = observations["agent-%i"%i]['cur_obs'] # --> field of vision of the ith agent

        action = agent.choose_action(observation) # should be ok
        actions["agent-%i"%i] = action
    new_observations, rewards, dones, infos = env.step(actions) #should be okay to

    return new_observations, rewards, dones, infos

def make_all_agents_learn(observations, new_observations, rewards, dones, infos):
    score_step = 0 # we need to return it and to add it to the function called make_all_agent_learn
    for i in range(len(observations)):
        reward = rewards["agent-%i"%i]
        score_step += reward

        observation = observations["agent-%i"%i]['cur_obs']
        new_observation = new_observations["agent-%i"%i]['cur_obs']
        done = dones["agent-%i"%i]
        agent = neuronal_agents[i]
        agent.learn(observation, reward, new_observation, done)
    return  new_observations

if __name__ == '__main__':
    # define all the parameters
    number_steps = 1000
    number_agents = 2
    number_games = 18
    alpha = 1e-5
    gamma = 0.99
    load_checkpoint = False


    env = HarvestEnv(num_agents=number_agents)
    A2C_agents = []


    for i in range(number_agents):
        A2C_agents.append(Agent(alpha=alpha, gamma=gamma))  # TODO tune the parameters of A2C agents


    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'harvest_%d_%d_%f_%f_%d_steps.png'%(number_agents, number_games, alpha, gamma, number_steps)
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        for i in range(number_agents):
            A2C_agents[i].load_models()

    for i in range(number_games):
        observations = env.reset()  # should be ok
        done = False
        score = 0
        # TODO handle the done
        while not done:

            make_all_agent_play(observations, neuronal_agents)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(number_games)]
        #plot_learning_curve(x, score_history, figure_file)
