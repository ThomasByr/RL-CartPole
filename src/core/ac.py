from typing import Tuple

import tensorflow as tf
from tensorflow.python.keras import layers


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")  # common network
        self.actor = layers.Dense(num_actions)  # actor network
        self.critic = layers.Dense(1)  # critic network

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
