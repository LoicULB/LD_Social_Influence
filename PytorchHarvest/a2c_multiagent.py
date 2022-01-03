import sys
import torch

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from a2c_models import ActorCritic


def make_all_agents_act(agent: ActorCritic, states, num_outputs, env, agents_log_probs,
                        agents_rewards, agents_values, entropies):
    agents_dict_actions = {}
    for i in range(len(states)):
        observation = states[f"agent-{i}"]['curr_obs']
        value, policy_dist = agent.forward(observation)
        value = value.detach().numpy()[0, 0]
        dist = policy_dist.detach().numpy()
        action, entropy, log_prob = before_env_step(dist, num_outputs, policy_dist)
        agents_dict_actions[f"agent-{i}"] = action

        #  update entropy
        entropies[i] += entropy

        # appends the values
        agents_log_probs[i].append(log_prob)
        agents_values[i].append(value)

    new_state_dic, reward_dic, done_dic, _ = env.step(agents_dict_actions)

    for i in range(len(states)):
        reward_i = reward_dic[f"agent-{i}"]
        agents_rewards[i].append(reward_i)

    return new_state_dic, reward_dic, done_dic


def a2c(env, nb_agents=4, GAMMA=0.99, num_steps=300, max_episodes=30, render_env=False, learning_rate=0.01,
        hidden_size=64):
    num_inputs = 675  # env.observation_space.shape[0] # is the dimension of the input (15*15*3)
    num_outputs = env.action_space.n

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    entropies = [0] * nb_agents
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        agents_log_probs = []
        append_empty_list_for_every_agents(agents_log_probs, nb_agents)

        agents_rewards = []
        append_empty_list_for_every_agents(agents_rewards, nb_agents)

        agents_values = []
        append_empty_list_for_every_agents(agents_values, nb_agents)

        dic_states = env.reset()

        for step in range(num_steps):
            new_state_dic, reward_dic, done_dic = make_all_agents_act(actor_critic, dic_states,
                                                                      num_outputs, env, agents_log_probs,
                                                                      agents_rewards, agents_values, entropies)

            #  entropy_term += entropy  # update entropy
            dic_states = new_state_dic

            if step == num_steps - 1:
                Qval = end_episode(actor_critic, all_lengths, all_rewards, average_lengths, new_state, rewards, step)
                if episode % 10 == 0:
                    print_episode_state(average_lengths, episode, rewards, step)
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
    smoothed_rewards = get_smoothed_rewards(all_rewards)

    plot_rewards_evolution(all_rewards, smoothed_rewards)


def append_empty_list_for_every_agents(list_to_append, nb_agents):
    for i in range(nb_agents):
        list_to_append.append([])


def before_env_step(dist, num_outputs, policy_dist):
    action = np.random.choice(num_outputs, p=np.squeeze(dist))
    log_prob = torch.log(policy_dist.squeeze(0)[action])
    entropy = -np.sum(np.mean(dist) * np.log(dist))
    return action, entropy, log_prob


def append_values(log_prob, log_probs, reward, rewards, value, values):
    rewards.append(reward)
    values.append(value)
    log_probs.append(log_prob)


def end_episode(actor_critic, all_lengths, all_rewards, average_lengths, new_state, rewards, steps):
    Qval, _ = actor_critic.forward(new_state)
    Qval = Qval.detach().numpy()[0, 0]
    all_rewards.append(np.sum(rewards))
    all_lengths.append(steps)
    average_lengths.append(np.mean(all_lengths[-10:]))
    return Qval


def print_episode_state(average_lengths, episode, rewards, steps):
    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
                                                                                               np.sum(
                                                                                                   rewards),
                                                                                               steps,
                                                                                               average_lengths[
                                                                                                   -1]))


def get_smoothed_rewards(all_rewards):
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    return smoothed_rewards


def plot_episode_length_evolution(all_lengths, average_lengths):
    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()


def plot_rewards_evolution(all_rewards, smoothed_rewards):
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
