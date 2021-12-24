import os
import tensorflow.keras
from tensorflow.keras.layers import Dense

class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512, name="actor_critic", chkpt_dir="tmp/actor_critic"):
        super(ActorCriticNetwork, self).__init__()
        self