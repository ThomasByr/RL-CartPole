import gym
from enum import Enum

__all__ = ["Config"]


class Config(Enum):
    """Enum for the different configurations."""

    cv0 = "CartPole-v0"  # CartPole v0
    cv1 = "CartPole-v1"  # CartPole v1

    def to_gym(self) -> gym.Env:
        """Convert the enum to a gym environment."""
        return gym.make(self.value)

    def to_str(self) -> str:
        """Convert the enum to a lowercased string."""
        return (_ := self.value.lower()).strip()

    def get_reward_threshold(self) -> int:
        """Get the reward threshold for the environment."""
        return {
            self.cv0: 195,
            self.cv1: 475,
        }[self]
