import numpy as np
from _neworder_core import MonteCarlo  # type: ignore[import]


def as_np(mc: MonteCarlo) -> np.random.Generator:
  class _NpAdapter(np.random.BitGenerator):
    def __init__(self, rng: MonteCarlo):
      super().__init__(0)
      self.rng = rng
      self.rng.init_bitgen(self.capsule)  # type: ignore

  return np.random.Generator(_NpAdapter(mc)) # type: ignore
