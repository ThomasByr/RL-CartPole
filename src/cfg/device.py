"""
Device

Device enum to force use with a specific device.
"""

# pylint: disable=[C0103, C0410, C0411]

from enum import Enum

import tensorflow as tf

__all__ = ["Device"]

N = len(tf.config.experimental.list_physical_devices("GPU"))


class Device(Enum):
  """Enum for the different devices."""

  cpu = "cpu"  # CPU
  gpu = "gpu"  # GPU
  auto = "auto"  # Auto

  def get_name(self) -> str:
    """Get the name of the device."""
    return {
        self.cpu: "/cpu:0",
        self.gpu: "/gpu:0",
        self.auto: "/gpu:0" if N > 0 else "/cpu:0",
    }[self]

  @staticmethod
  def nd() -> int:
    """Get the number of available devices."""
    return N
