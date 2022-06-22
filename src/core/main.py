"""
Main entry point.
"""
# pylint: disable=[C0410, C0410, C0411, W0401, W0614]

from typing import List

from .env import Env
from ..cfg import *

from ..__version__ import __version__, __title__

import getopt, sys

__all__ = ["main"]


def get_help() -> None:  # pragma: no cover
    """Display the help message."""
    info("Usage: python main.py [-hvc] [--help] [--version] [--cfg <cfg>] [--cpu] [--gpu] [--auto]")
    info("")
    info("Options:")
    info("  -h, --help      Show this help message and exit.")
    info("  -v, --version   Show version and exit.")
    info("  -c, --cfg <cfg> Set the CartPole configuration version (default: {}).", Config.cv0)
    info("                  Available versions from \"cv0\" or \"cv1\".")
    info("  --cpu           Force use of the CPU.")
    info("  --gpu           Force use of the GPU.")
    info("  --auto          Use the GPU if available, otherwise the CPU.")


def get_version() -> None:  # pragma: no cover
    """Display the version of the program."""
    info("%s v%s", __title__, __version__)


def main(argv: List[str]) -> None:
    """
    Main entry point.

    ## Parameters
    ```py
    argv: List[str]
        Command line arguments
        (tries to strip the program name from the list).
    ```
    """
    try:
        opts, args = getopt.getopt(
            list(map(str.lower, argv[::] if "main.py" not in argv[0] else argv[1:])),
            "hvc:",
            [
                "help",
                "version",
                "config=",
                "cfg=",
                "cpu",
                "gpu",
                "auto",
            ],
        )
    except getopt.GetoptError:
        error("Unrecognized option.")
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
    use_auto = False
    use_cpu, use_gpu = False, False  # keep track of the force use of the CPU or GPU
    for opt, arg in opts:
        if opt in ("-c", "--config", "--cfg"):
            kwargs["cfg"] = arg
        elif opt == "--cpu":
            kwargs["device"] = Device.cpu
            use_cpu = True
        elif opt == "--gpu":
            kwargs["device"] = Device.gpu
            use_gpu = True
        elif opt == "--auto":
            kwargs["device"] = Device.auto
            use_auto = True

    if use_cpu and use_gpu:
        error("Only one device can be used at a time.")
        get_help()
        sys.exit(1)
    elif use_auto:
        if use_gpu or use_cpu:
            error("The --auto option cannot be used with --cpu or --gpu.")
            get_help()
            sys.exit(1)

    environment = Env(*args, **kwargs)
    environment.run()
