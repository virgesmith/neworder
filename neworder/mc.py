import numpy as np

from neworder import MonteCarlo


def as_np(mc: MonteCarlo) -> np.random.Generator:
    """
    Returns an adapter enabling the MonteCarlo object to be used with numpy random functionality
    """

    class _NpAdapter(np.random.BitGenerator):
        def __init__(self, rng: MonteCarlo):
            super().__init__(0)
            self.rng = rng
            self.rng.init_bitgen(self.capsule)

    return np.random.Generator(_NpAdapter(mc))
