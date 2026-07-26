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

    If group_transition_matrices is supplied, the population is additionally split into groups
    (e.g. "fast"/"slow" movers), each following its own transition matrix via
    `no.df.transition_conditional`, tracked alongside (but independently of) the single-matrix
    `no.df.transition` result above - see `mixed_stationary_distribution()` for why the two can
    reach different equilibria even though every individual's transitions follow the same rules
    either way.
    """

    def __init__(
        self,
        timeline: no.Timeline,
        npeople: int,
        states: npt.NDArray[np.int64],
        transition_matrix: npt.NDArray[np.float64],
        use_python_impl: bool = False,
        group_transition_matrices: dict[str, npt.NDArray[np.float64]] | None = None,
    ) -> None:
        super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)

        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("each row of the transition matrix must sum to 1")
        if group_transition_matrices is not None:
            for group, matrix in group_transition_matrices.items():
                if not np.allclose(matrix.sum(axis=1), 1.0):
                    raise ValueError(f"each row of the '{group}' transition matrix must sum to 1")

        self.npeople = npeople
        self.states = states
        self.transition_matrix = transition_matrix
        self.use_python_impl = use_python_impl
        self.group_transition_matrices = group_transition_matrices

        self.pop = pd.DataFrame(data={"state": pd.Categorical(np.zeros(npeople, dtype=np.int64), categories=states)})
        if group_transition_matrices is not None:
            groups = list(group_transition_matrices)
            self.pop["group"] = pd.Categorical([groups[i % len(groups)] for i in range(npeople)], categories=groups)
            self.pop["state_conditional"] = self.pop["state"]

        self.summary = pd.DataFrame(columns=states, dtype=np.int64)
        self.summary.loc[0] = self._state_counts("state")

        if group_transition_matrices is not None:
            self.summary_conditional = pd.DataFrame(columns=states, dtype=np.int64)
            self.summary_conditional.loc[0] = self._state_counts("state_conditional")

        # cumulative time spent in the transition step, for comparing the python/C++ implementations
        self.transition_time_s = 0.0

    def _state_counts(self, colname: str) -> pd.Series:
        return self.pop[colname].value_counts().reindex(self.states, fill_value=0)

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
            self.pop["state"] = no.df.transition(self.mc, self.transition_matrix, self.pop["state"])
        self.transition_time_s += time.perf_counter() - t0

        self.summary.loc[len(self.summary)] = self._state_counts("state")

        if self.group_transition_matrices is not None:
            self.pop["state_conditional"] = no.df.transition_conditional(
                self.mc, self.group_transition_matrices, self.pop["group"], self.pop["state_conditional"]
            )
            self.summary_conditional.loc[len(self.summary_conditional)] = self._state_counts("state_conditional")

    def finalise(self) -> None:
        self.summary["t"] = np.arange(self.timeline.start, self.timeline.end + 1e-8, self.timeline.dt)
        self.summary.reset_index(drop=True, inplace=True)

        if self.group_transition_matrices is not None:
            self.summary_conditional["t"] = self.summary["t"]
            self.summary_conditional.reset_index(drop=True, inplace=True)

        impl = "python" if self.use_python_impl else "C++"
        no.log(f"{impl} transition implementation: {self.transition_time_s:.2f}s over {len(self.summary) - 1} steps")

    @staticmethod
    def _stationary_distribution(transition_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        pi = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        return pi / pi.sum()

    def stationary_distribution(self) -> npt.NDArray[np.float64]:
        """
        Computes the analytic equilibrium (stationary) distribution of the Markov chain - i.e. the
        left eigenvector of the transition matrix corresponding to eigenvalue 1 - for comparison
        against the simulated results.
        """
        return self._stationary_distribution(self.transition_matrix)

    def mixed_stationary_distribution(self) -> npt.NDArray[np.float64]:
        """
        Computes the population-share-weighted average of each group's own analytic stationary
        distribution. Since group membership never changes, this - not stationary_distribution(),
        which assumes everyone follows the single pooled transition_matrix - is what the population
        actually converges to under transition_conditional whenever the groups' dynamics differ.
        """
        assert self.group_transition_matrices is not None
        weights = self.pop["group"].value_counts(normalize=True)
        mixed = np.zeros(len(self.states))
        for group, matrix in self.group_transition_matrices.items():
            mixed += weights[group] * self._stationary_distribution(matrix)
        return mixed
