import pytest

from ..core import Env


@pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
def test_v0():
    """Test the train function."""
    environment = Env("0")
    environment.run()


@pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
def test_v1():
    """Test the train function."""
    environment = Env("1")
    environment.run()
