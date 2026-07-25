import time

import numpy as np
import numpy.typing as npt
import pandas as pd

import neworder as no


class MarkovChain(no.Model):
    """
    Simulates a population of individuals, each independently transitioning between discrete
    states according to a Markov process, and tracks the size of each state's population over
    time.

    Also implements a pure-python equivalent of `no.df.transition` (see `transition_py` below),
    to illustrate the performance gain of neworder's C++ implementation - see `step()`.
    """

    def __init__(
        self,
        timeline: no.Timeline,
        npeople: int,
        states: npt.NDArray[np.int64],
        transition_matrix: npt.NDArray[np.float64],
        use_python_impl: bool = False,
    ) -> None:
        super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)

        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("each row of the transition matrix must sum to 1")

        self.npeople = npeople
        self.states = states
        self.transition_matrix = transition_matrix
        self.use_python_impl = use_python_impl

        self.pop = pd.DataFrame(data={"state": pd.Categorical(np.zeros(npeople, dtype=np.int64), categories=states)})

        self.summary = pd.DataFrame(columns=states, dtype=np.int64)
        self.summary.loc[0] = self._state_counts()

        # cumulative time spent in the transition step, for comparing the python/C++ implementations
        self.transition_time_s = 0.0

    def _state_counts(self) -> pd.Series:
        return self.pop.state.value_counts().reindex(self.states, fill_value=0)

    def transition_py(self, colname: str) -> None:
        """Pure-python equivalent of `no.df.transition`, for performance comparison."""

        def _interp(cumprob: npt.NDArray[np.float64], x: float) -> int:
            lbound = 0
            while lbound < len(cumprob) - 1 and cumprob[lbound] <= x:
                lbound += 1
            return lbound

        cumprob = np.cumsum(self.transition_matrix, axis=1)

        # codes are already indices 0..m-1 into self.states, so (like the C++ implementation) no value -> index
        # lookup is needed
        u = self.mc.ustream(len(self.pop))
        codes = self.pop[colname].cat.codes.to_numpy()
        new_codes = np.array([_interp(cumprob[c], ui) for c, ui in zip(codes, u, strict=True)])
        self.pop[colname] = pd.Categorical.from_codes(new_codes, categories=pd.Index(self.states))

    def step(self) -> None:
        t0 = time.perf_counter()
        if self.use_python_impl:
            self.transition_py("state")
        else:
            self.pop["state"] = no.df.transition(self, self.transition_matrix, self.pop, "state")
        self.transition_time_s += time.perf_counter() - t0

        self.summary.loc[len(self.summary)] = self._state_counts()

    def finalise(self) -> None:
        self.summary["t"] = np.arange(self.timeline.start, self.timeline.end + 1e-8, self.timeline.dt)
        self.summary.reset_index(drop=True, inplace=True)

        impl = "python" if self.use_python_impl else "C++"
        no.log(f"{impl} transition implementation: {self.transition_time_s:.2f}s over {len(self.summary) - 1} steps")

    def stationary_distribution(self) -> npt.NDArray[np.float64]:
        """
        Computes the analytic equilibrium (stationary) distribution of the Markov chain - i.e. the
        left eigenvector of the transition matrix corresponding to eigenvalue 1 - for comparison
        against the simulated results.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        pi = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        return pi / pi.sum()
