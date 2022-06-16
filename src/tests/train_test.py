import pytest
import tensorflow as tf
import numpy as np

from ..core import Env, ActorCritic


@pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
def test_v0():
    """Test the train function."""
    environment = Env("0")

    seed = 42
    environment.env.reset(seed=seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_actions = environment.env.action_space.n

    num_hidden_units = 1 << 7

    model = ActorCritic(num_actions, num_hidden_units)
    environment.train(model)


@pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
def test_v1():
    """Test the train function."""
    environment = Env("1")

    seed = 42
    environment.env.reset(seed=seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_actions = environment.env.action_space.n

    num_hidden_units = 1 << 7

    model = ActorCritic(num_actions, num_hidden_units)
    environment.train(model)
