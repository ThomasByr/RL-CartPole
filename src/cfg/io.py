"""
I/O utility functions

Log level functions.
"""

# pylint: disable=[C0103, C0410, C0411]

from typing import NoReturn
from termcolor import colored

import re, sys

__all__ = ["info", "debug", "warn", "error", "fatal"]


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
