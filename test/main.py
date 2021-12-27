import gym
import numpy as np
from ActorCriticAgent import Agent
from harvest import HarvestEnv
from utils import plot_learning_curve
from gym import wrappers

def make_all_agent_play(observations, neuronal_agents : Agent ):
    actions = {}
    
    for i in range(len(observations)):
        observation = observations[f"agent-{i}"]['cur_obs']
        agent = neuronal_agents[i]
        
        action = agent.choose_action(observation) # should be ok
        actions[f"agent-{i}"] = action
    new_observations, rewards, dones, infos = env.step(actions) #should be okay to
    return   new_observations, rewards, dones, infos  
def make_all_agents_learn(new_observation, rewards, dones, infos):    
    score_step = 0 # we need to return it and to add it to the function called make_all_agent_learn
    for i in range(len(observations)):
        reward = rewards[f"agent-{i}"]
        score_step += reward
        
        if not load_checkpoint:
            observation = observations[f"agent-{i}"]['cur_obs']
            new_observation = new_observations[f"agent-{i}"]['cur_obs']
            done = dones[f"agent-{i}"]
            agent = neuronal_agents[i]
            
            agent.learn(observation, reward, new_observation, done)
    return  new_observations
    
if __name__ == '__main__':
    env = HarvestEnv(num_agents=2)
    
    agent = Agent(alpha=1e-5, ) # TODO define list of neunal agents
    n_games = 18
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'cartpole_1e-5_1024x512_1800games.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observations = env.reset() # should be ok
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
        x = [i+1 for i in range(n_games)]
        #plot_learning_curve(x, score_history, figure_file)