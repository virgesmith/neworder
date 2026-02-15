import time
from threading import Barrier, Lock

import pandas as pd

import neworder


class ParallelThreaded(neworder.Model):
    """
    A simple model demonstrating multithreaded rather than multiprocess execution, sharing a dataset and synchronising
    at each timestep. An expensive computation is simulated that is O(n), showing how partitioning the dataset can
    achieve significant performance gains.
    """

    def __init__(self, pop: pd.DataFrame, indices: list[int], lock: Lock, barrier: Barrier) -> None:
        """
        pop: the overall population
        indices: the chunk of the population that this thread deals with
        """
        super().__init__(neworder.LinearTimeline(0, 10, 10), lambda: indices[0])  # deterministic, independent
        self.lock = lock
        self.barrier = barrier
        self.pop = pop  # ref not copy
        self.indices = indices

    def step(self) -> None:
        # simulate some complex CPU-bound task that scales linearly
        time.sleep(len(self.indices) / len(self.pop))
        value = self.mc.ustream(1)

        # update the dataset
        with self.lock:  # throwing in a lock is bad
            self.pop.loc[self.indices] += value
        # (optional) synchronise threads at the end of each timestep
        self.barrier.wait()
