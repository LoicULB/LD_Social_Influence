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

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 1000


def a2c(env):
    num_inputs = 675  # env.observation_space["curr_obs"].shape[0]
    num_outputs = env.action_space.n

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        state = state["agent-0"]["curr_obs"]

        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action, entropy, log_prob = before_env_step(dist, num_outputs, policy_dist)
            action_dic = {"agent-0": action}
            new_states, reward_dic, dones, _ = env.step(action_dic)

            # extract from the dic
            new_state = new_states["agent-0"]["curr_obs"]
            reward = reward_dic["agent-0"]
            done = dones["agent-0"]

            append_values(log_prob, log_probs, reward, rewards, value, values)

            entropy_term += entropy  # update entropy
            state = new_state

            if done or steps == num_steps - 1:
                Qval = end_episode(actor_critic, all_lengths, all_rewards, average_lengths, new_state, rewards, steps)
                if episode % 10 == 0:
                    print_episode_state(average_lengths, episode, rewards, steps)
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
    plot_episode_length_evolution(all_lengths, average_lengths)


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