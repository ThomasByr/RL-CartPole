from typing import List

from .env import Env
from ..cfg import *

from ..__version__ import __version__

import getopt

__all__ = ["main"]


def get_help() -> None:  # pragma: no cover
    info("Usage: python3 main.py [-hv] [--help] [--version=<version>]")
    info("")
    info("Options:")
    info("  -h, --help      Show this help message and exit.")
    info("  -v, --version   Show version and exit.")
    info("  --config=<cfg>  Set the CartPole configuration version (default: %s).", Config.cv0)
    info("                  Available versions from \"cv0\" or \"cv1\".")


def get_version() -> None:  # pragma: no cover
    info("Version: %s", __version__)


def main(argv: List[str]) -> None:
    try:
        opts, args = getopt.getopt(argv[1:], "hvc:", ["help", "version", "config="])
    except getopt.GetoptError:
        get_help()
        exit()
    finally:
        for opt, _ in opts:
            if opt in ("-h", "--help"):
                get_help()
                exit()
            elif opt in ("-v", "--version"):
                get_version()
                exit()

    kwargs = {}
    for opt, arg in opts:
        if opt in ("-c", "--config"):
            kwargs["cfg"] = arg

    environment = Env(*args, **kwargs)
    environment.run()
