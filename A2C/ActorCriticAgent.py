import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from A2C.ActorCriticModel import ActorCriticNetwork

from social_dilemma.environment.agent import HARVEST_ACTIONS

class Agent():
    """
    A2C Agent
    """
    def __init__(self, 
                 alpha=0.0003, gamma=0.99
                 ):
        
        self.gamma = gamma
        self.n_actions = len(HARVEST_ACTIONS)
        self.action = None  # TODO to handle
        self.action_space = list(HARVEST_ACTIONS.keys())

        self.actor_critic = ActorCriticNetwork(n_actions=self.n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))


    def choose_action(self, observation):
        """
        Choose the action with highest estimated probability by the A2C neuronal network
        :param observation: the observation of the agent
        :return: the action of the agent (between 0 and 7)
        """
        observation = observation.flatten()
        
        state = tf.convert_to_tensor([observation]) 
        _, probs = self.actor_critic(state)
        
        proba = probs.numpy()[0]
        action = proba.argmax()
        self.action = action

        return action

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
        
    def learn(self, state, reward, state_, done):
        # convert state to a single dimension array
        state = np.array(state)
        state = state.flatten()
        state_ = np.array(state_)
        state_ = state_.flatten()

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        # state_ is the new state.
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))