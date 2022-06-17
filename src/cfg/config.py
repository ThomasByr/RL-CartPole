from enum import Enum
from typing import NoReturn

import gym
from termcolor import colored

__all__ = ["Config", "info", "debug", "warn", "error", "fatal"]


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


def info(msg: str, *args: object) -> None:
    """Print a message to stdout to the info level."""
    print(colored(msg % args, "blue"))


def debug(msg: str, *args: object) -> None:
    """Print a message to stdout to the debug level."""
    print(colored(msg % args, "cyan"))


def warn(msg: str, *args: object) -> None:
    """Print a message to stdout to the warn level."""
    print(colored(msg % args, "yellow"))


def error(msg: str, *args: object) -> None:
    """Print a message to stdout to the error level."""
    print(colored(msg % args, "red"))


def fatal(msg: str, *args: object) -> NoReturn:
    """Print a message to stdout to the fatal level and exit."""
    print(colored(msg % args, "red"))
    exit(1)
