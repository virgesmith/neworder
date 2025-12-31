from concurrent.futures import ThreadPoolExecutor
from threading import get_native_id
from typing import Callable

import numpy as np
import numpy.typing as npt

import neworder as no

N_THREADS = 4


def test_thread_id() -> None:
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda: get_native_id() == no.thread_id()) for i in range(N_THREADS)]
        assert all(f.result() for f in futures)


def test_unique_index() -> None:
    n1 = 10
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda: no.df.unique_index(n1)) for i in range(N_THREADS)]
        result = np.concat([f.result() for f in futures])
    assert len(set(result)) == n1 * N_THREADS

    if no.mpi.SIZE > 1:
        all_results = no.mpi.COMM.gather(result, root=0)
        if all_results is not None:
            result = np.concat(all_results)
            assert len(set(result)) == n1 * N_THREADS * no.mpi.SIZE, len(set(result))


def test_rng_independence() -> None:
    class TestModel(no.Model):
        def __init__(self, n: int, seeder: Callable[[], int]) -> None:
            super().__init__(no.NoTimeline(), seeder)
            self.n = n

        def step(self) -> None:
            self.stream = self.mc.ustream(self.n)

    def run_model(n: int, seeder: Callable[[], int]) -> npt.NDArray[np.float64]:
        m = TestModel(n, seeder)
        no.run(m)
        return m.stream

    # identical streams
    N = 100
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(lambda: run_model(N, no.MonteCarlo.deterministic_identical_stream)) for _ in range(2)
        ]
        result = [f.result() for f in futures]
        assert (result[0] == result[1]).all()

    # independent streams
    N = 1_000_000
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda i=i: run_model(N, lambda: i)) for i in range(2)]
        result = [f.result() for f in futures]
        rho = np.corrcoef(*result)[0, 1]
        assert abs(rho) < 2 / np.sqrt(N)


if __name__ == "__main__":
    test_rng_independence()
