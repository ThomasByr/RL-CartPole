import pytest
from termcolor import colored

from ..main import ActorCritic, train, env


@pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
def test_train():
    num_actions = env.action_space.n  # 2

    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    assert model is not None

    try:
        train(model)
    except Exception as e:
        print(colored(f"Error: {e}", "red"))
        assert False

    assert True
