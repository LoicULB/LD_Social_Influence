import os

import numpy as np
from A2C.ActorCriticAgent import Agent
from social_dilemma.environment.harvest import HarvestEnv
from run.utils import plot_learning_curve
from run.utils import plot_curve
from numpy import asarray
import numpy as np


def make_all_agent_play(observations, A2C_agents): # : Agent
    actions = {}

    for i in range(len(A2C_agents)):
        agent = A2C_agents[i]                                   # --> agent i
        observation = observations["agent-%d"%i]['curr_obs']     # --> field of vision of the ith agent

        action = agent.choose_action(observation)  # should be ok
        actions["agent-%d"%i] = action
    new_observations, rewards, dones, infos = env.step(actions)  # should be okay to

    return new_observations, rewards, dones, infos

def create_agents(alpha, gamma, number_agents):
    A2C_agents = []
    for game in range(number_agents):
        A2C_agents.append(Agent(alpha=alpha, gamma=gamma))  # TODO tune the parameters of A2C agents
    return A2C_agents

def make_all_agents_learn(observations, new_observations, rewards, dones, infos, A2C_agents):
    score_step = 0  # we need to return it and to add it to the function called make_all_agent_learn
    for i in range(len(A2C_agents)):
        reward = rewards["agent-%d"%i]
        score_step += reward

        observation = observations["agent-%d"%i]['curr_obs']
        new_observation = new_observations["agent-%d"%i]['curr_obs']
        done = dones["agent-%d"%i]
        agent = A2C_agents[i]
        agent.learn(observation, reward, new_observation, done)
    return score_step

if __name__ == '__main__':
    # TODO render how the agents act

    # define all the parameters
    number_steps = 100
    number_agents = 2
    number_games = 2
    alpha = 1e-5
    gamma = 0.99
    load_checkpoint = False


    env = HarvestEnv(num_agents=number_agents)

    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #environment = wrappers.Monitor(environment, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'harvest_%d_%d_%f_%f_%d_steps.png'%(number_agents, number_games, alpha, gamma, number_steps)
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    games_score_history = []

    """
    A2C_agents = []
    A2C_agents = create_agents(alpha, gamma, number_agents)
    if load_checkpoint:
        for i in range(number_agents):
            A2C_agents[i].load_models()
    """

    games_score_steps = []
    #else: # training
    for game in range(number_games):

        observations = env.reset()  # should be ok
        A2C_agents = create_agents(alpha, gamma, number_agents)
        accumulative_collective_score = 0
        collective_score_step = []
        for step in range(0, number_steps):
            new_observations, rewards, dones, infos = make_all_agent_play(observations, A2C_agents)
            collective_score_step.append(make_all_agents_learn(observations, new_observations, rewards, dones, infos, A2C_agents))
            # update
            observation = new_observations
            accumulative_collective_score += collective_score_step[step]

        # a game is finish here
        # save data
        games_score_steps.append(collective_score_step)
        avg_score = np.mean(collective_score_step[-min(100, number_steps):])
        #avg_score = np.mean(collective_score_step)
        games_score_history.append(accumulative_collective_score)

        print('game ', game, 'score %.1f' % accumulative_collective_score, 'avg_score %.1f' % avg_score)

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                print('... saving models ...')
                for agent in range(number_agents):
                    A2C_agents[agent].save_models()


    total_games_scores = asarray(games_score_steps)
    mean = np.mean(total_games_scores, axis=0)
    name_curve = "collective mean reward"
    fig_file_test = os.path.join(os.getcwd(), "plots", name_curve)
    plot_curve(games_score_steps, fig_file_test)

    """
    if not load_checkpoint:
        x = [i + 1 for i in range(number_games)]
        plot_learning_curve(x, games_score_history, figure_file)
    """
