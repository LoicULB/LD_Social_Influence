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
from Hyperparams import get_first_batch
from Hyperparams import get_papers_batch, get_third_batch
from collections import Counter
# hyperparameters




def a2c(env, GAMMA=0.99, num_steps=300, max_episodes=30, render_env = False, learning_rate=0.00136, hidden_size=18):
    num_inputs = 675  # env.observation_space["curr_obs"].shape[0] # is the dimension of the input (15*15*3) = 675
    num_outputs = env.action_space.n

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    history_rewards = []
    full_actions_history = []
    for episode in range(max_episodes):
        actions_history = []

        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        state = state["agent-0"]["curr_obs"]

        if render_env and episode == 29:
            env.render('tmp/img/harvest_initial_step')
        for step in range(num_steps):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action, entropy, log_prob = before_env_step(dist, num_outputs, policy_dist)
            actions_history.append(action)
            action_dic = {"agent-0": action}
            new_states, reward_dic, dones, _ = env.step(action_dic)

            if render_env and episode == 29:
                env.render('tmp/img/harvest_step_%d' % (step))

            # extract from the dic
            new_state = new_states["agent-0"]["curr_obs"]
            reward = reward_dic["agent-0"]
            done = dones["agent-0"]

            append_values(log_prob, log_probs, reward, rewards, value, values)

            entropy_term += entropy  # update entropy
            state = new_state

            if done or step == num_steps - 1:
                Qval = end_episode(actor_critic, all_lengths, all_rewards, average_lengths, new_state, rewards, step)
                if episode % 5 == 0:  # TODO it was 10
                    current_episode_actions = Counter(actions_history)
                    full_actions_history.append(current_episode_actions)
                    # print(current_episode_actions)
                    # print_episode_state(episode, rewards, history_rewards)
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

    sum_rewards = np.sum(all_rewards)

    #print("all_rewards: ", all_rewards)
    #print("sum all_rewards: ", sum_rewards)

    # Plot results
    smoothed_rewards = get_smoothed_rewards(all_rewards)
    return history_rewards, sum_rewards
    #plot_rewards_evolution(all_rewards, smoothed_rewards)
    #plot_episode_length_evolution(all_lengths, average_lengths)


def before_env_step(dist, num_outputs, policy_dist):
    action = np.random.choice(num_outputs, p=np.squeeze(dist))
    log_prob = torch.log(policy_dist.squeeze(0)[action])
    #could resolve the log divided by zero error
    #dist = np.where(dist > 0.0000000001, dist, 0.00000001)
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


def print_episode_state(episode, rewards, history_rewards):
    rewards_episode = np.sum( rewards)
    history_rewards.append(rewards_episode)
    sys.stdout.write("episode: {}, reward: {}\n".format(episode,   rewards_episode ))


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