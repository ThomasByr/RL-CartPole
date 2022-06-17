"""
Main entry point.
"""
#pylint: disable=[C0410, C0410, C0411, W0401, W0614]

from typing import List

from .env import Env
from ..cfg import *

from ..__version__ import __version__, __title__

import getopt, sys

__all__ = ["main"]


def get_help() -> None:  # pragma: no cover
    """Display the help message."""
    info("Usage: python3 main.py [-hvc] [--help] [--version] [--config=<cfg>]")
    info("")
    info("Options:")
    info("  -h, --help      Show this help message and exit.")
    info("  -v, --version   Show version and exit.")
    info("  --config=<cfg>  Set the CartPole configuration version (default: %s).", Config.cv0)
    info("                  Available versions from \"cv0\" or \"cv1\".")


def get_version() -> None:  # pragma: no cover
    """Display the version of the program."""
    info("%s v%s", __title__, __version__)


def main(argv: List[str]) -> None:
    """Main entry point."""
    try:
        opts, args = getopt.getopt(argv[1:], "hvc:", ["help", "version", "config="])
    except getopt.GetoptError:
        get_help()
        sys.exit(1)
    finally:
        for opt, _ in opts:
            if opt in ("-h", "--help"):
                get_help()
                sys.exit()
            elif opt in ("-v", "--version"):
                get_version()
                sys.exit()

    kwargs = {}
    for opt, arg in opts:
        if opt in ("-c", "--config"):
            kwargs["cfg"] = arg

    environment = Env(*args, **kwargs)
    environment.run()
