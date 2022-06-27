"""
Config and utility functions

Config enum and log level functions.
"""

# pylint: disable=[C0103, C0410, C0411]

from enum import Enum
from typing import NoReturn
from termcolor import colored

import re, gym, sys

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

  def get_name(self) -> str:
    """Get the name of the environment."""
    return {
        self.cv0: "CartPole-v0",
        self.cv1: "CartPole-v1",
    }[self]

  def get_reward_threshold(self) -> int:
    """Get the reward threshold for the environment."""
    return {
        self.cv0: 195,
        self.cv1: 475,
    }[self]


def _format_msg(msg: str, *args: object) -> str:
  """Format a message."""
  return msg % args if re.search(r"%[sdf]|%.[0-9]f", msg) else msg.format(*args)


def info(msg: str, *args: object) -> None:
  """Print a message to stdout to the info level."""
  print(colored("   INFO", "green"), _format_msg(msg, *args))


def debug(msg: str, *args: object) -> None:
  """Print a message to stdout to the debug level."""
  print(colored("  DEBUG", "cyan"), _format_msg(msg, *args))


def warn(msg: str, *args: object) -> None:
  """Print a message to stdout to the warn level."""
  print(colored("   WARN", "yellow"), _format_msg(msg, *args))


def error(msg: str, *args: object) -> None:
  """Print a message to stdout to the error level."""
  print(colored("  ERROR", "red"), _format_msg(msg, *args))


def fatal(msg: str, *args: object) -> NoReturn:
  """Print a message to stdout to the fatal level and exit."""
  print(colored("  FATAL", "red"), _format_msg(msg, *args))
  sys.exit(1)
