import sys
from typing import List

import torch

import numpy as np
import torch.optim as optim

import matplotlib.pyplot as plt
import pandas as pd
from PytorchHarvest.a2c_models import ActorCritic


def make_all_agents_act(agents: List[ActorCritic], states, num_outputs, env, agents_log_probs,
                        agents_rewards, agents_values, entropies):
    agents_dict_actions = {}
    for i in range(len(states)):
        observation = states[f"agent-{i}"]['curr_obs']
        value, policy_dist = agents[i].forward(observation)
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


def a2c(env, nb_agents=4, GAMMA=0.99, num_steps=1000, max_episodes=30, render_env=False, learning_rate=0.01,
        hidden_size=64):
    num_inputs = 675  # env.observation_space.shape[0] # is the dimension of the input (15*15*3)
    num_outputs = env.action_space.n

    actor_critic_list = []
    optimizer_list = []
    for i in range(nb_agents):
        actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
        ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
        actor_critic_list.append(actor_critic)
        optimizer_list.append(ac_optimizer)

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
        if render_env and episode == 29:
            env.render('tmp/img/harvest_initial_step')
        for step in range(num_steps):
            new_state_dic, reward_dic, done_dic = make_all_agents_act(actor_critic_list, dic_states,
                                                                      num_outputs, env, agents_log_probs,
                                                                      agents_rewards, agents_values, entropies)

            #  entropy_term += entropy  # update entropy
            dic_states = new_state_dic
            if render_env and episode == 29:
                env.render('tmp/img/harvest_step_%d' % (step))
            if step == num_steps - 1:
                Qvals_agent = end_episode(actor_critic_list, new_state_dic)
                #if episode % 10 == 0:
                    #  print_episode_state(average_lengths, episode, rewards, step)
                break

        make_all_agents_learn(GAMMA, optimizer_list, entropies, Qvals_agent, agents_values, agents_rewards, agents_log_probs)

    # Plot results
    #smoothed_rewards = get_smoothed_rewards(all_rewards)

    #plot_rewards_evolution(all_rewards, smoothed_rewards)


def make_all_agents_learn(GAMMA, ac_optimizer_list, entropies, Qvals_agent, agents_values, agents_rewards, agents_log_probs):
    for i in range (len(agents_values)):
        rewards = agents_rewards[i]
        values = agents_values[i]
        Qval = Qvals_agent[i]
        log_probs = agents_log_probs[i]
        # compute Q values
        entropy_term = entropies[i]
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
        # update actor critic
        valuesTorch = torch.FloatTensor(values)
        QvalsTorch = torch.FloatTensor(Qvals)
        log_probsTorch = torch.stack(log_probs)
        advantage = QvalsTorch - valuesTorch
        actor_loss = (-log_probsTorch * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
        ac_optimizer_list[i].zero_grad()
        ac_loss.backward(retain_graph=True)
        ac_optimizer_list[i].step()


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


def end_episode(actor_critic_list , new_state):
    Qvals = []
    for i in range(len(new_state)):
        state = new_state[f"agent-{i}"]["curr_obs"]
        Qval, _ = actor_critic_list[i].forward(state)
        Qval = Qval.detach().numpy()[0, 0]
        Qvals.append(Qval)
    return Qvals


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
