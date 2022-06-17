"""
Actor Critic

Tensorflow (keras) model for the actor critic algorithm.
"""
#pylint: disable=[C0103, R0903, E0611]

from typing import Tuple

import tensorflow as tf
from tensorflow.python.keras import layers

__all__ = ["ActorCritic"]


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")  # common network
        self.actor = layers.Dense(num_actions)  # actor network
        self.critic = layers.Dense(1)  # critic network

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Call the model."""
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
