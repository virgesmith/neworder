"""
Run script for multithreaded parallel example
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pandas as pd
from parallel_mt import ParallelThreaded

import neworder

# neworder.verbose()

N_THREADS = 8
N_PEOPLE = 100

pop = pd.DataFrame(index=range(N_PEOPLE), data={"state": 0.0})


def sync() -> None:
    """Optionally add code here to run before the barrier is crossed"""
    pass


# threading primitives
lock = Lock()
barrier = Barrier(N_THREADS, action=sync)


def run_thread(index: int, lock: Lock) -> None:
    indices = list(range(index * N_PEOPLE // N_THREADS, (index + 1) * N_PEOPLE // N_THREADS))
    m = ParallelThreaded(pop, indices, lock, barrier)
    neworder.run(m)


neworder.log(f"neworder FT: {neworder.freethreaded()}, python FT:{not sys._is_gil_enabled()}", sys.version)


t = time.perf_counter()
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_thread, i, lock) for i in range(N_THREADS)]
    results = [f.result() for f in futures]
neworder.log(f"Execution time: {time.perf_counter() - t:.3f}s, (total CPU time ~ 10s)")

# This should output N_THREADS unique values, with (approximately) even counts
print(pop.value_counts())
