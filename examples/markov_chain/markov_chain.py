import time

import numpy as np
import numpy.typing as npt
import pandas as pd

import neworder as no


class MarkovChainBase(no.Model):
    """
    Shared scaffolding for the pooled-matrix (`MarkovChain`) and per-group (`ConditionalMarkovChain`)
    simulations below: population setup, state-count tracking, and the final "t" column/index
    cleanup. Each subclass provides its own `step()` (i.e. its own choice of transition matrix or
    matrices) and its own matrix validation.
    """

    def __init__(self, timeline: no.Timeline, npeople: int, states: npt.NDArray[np.int64]) -> None:
        super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)

        self.npeople = npeople
        self.states = states
        self.pop = pd.DataFrame(data={"state": pd.Categorical(np.zeros(npeople, dtype=np.int64), categories=states)})

        self.summary = pd.DataFrame(columns=states, dtype=np.int64)
        self.summary.loc[0] = self._state_counts()

    def _state_counts(self) -> pd.Series:
        return self.pop.state.value_counts().reindex(self.states, fill_value=0)

    def finalise(self) -> None:
        self.summary["t"] = np.arange(self.timeline.start, self.timeline.end + 1e-8, self.timeline.dt)
        self.summary.reset_index(drop=True, inplace=True)

    @staticmethod
    def _stationary_distribution(transition_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        pi = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        return pi / pi.sum()


class MarkovChain(MarkovChainBase):
    """
    Simulates a population of individuals, each independently transitioning between discrete
    states according to a single, population-wide Markov transition matrix, and tracks the size
    of each state's population over time.

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
        super().__init__(timeline, npeople, states)

        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("each row of the transition matrix must sum to 1")

        self.transition_matrix = transition_matrix
        self.use_python_impl = use_python_impl

        # cumulative time spent in the transition step, for comparing the python/C++ implementations
        self.transition_time_s = 0.0

    def transition_py(self) -> None:
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
        codes = self.pop["state"].cat.codes.to_numpy()
        new_codes = np.array([_interp(cumprob[c], ui) for c, ui in zip(codes, u, strict=True)])
        self.pop["state"] = pd.Categorical.from_codes(new_codes, categories=pd.Index(self.states))

    def step(self) -> None:
        t0 = time.perf_counter()
        if self.use_python_impl:
            self.transition_py()
        else:
            self.pop["state"] = no.df.transition(self.mc, self.transition_matrix, self.pop["state"])
        self.transition_time_s += time.perf_counter() - t0

        self.summary.loc[len(self.summary)] = self._state_counts()

    def finalise(self) -> None:
        super().finalise()

        impl = "python" if self.use_python_impl else "C++"
        no.log(
            f"{impl} transition implementation ({self.__class__.__name__}): {self.transition_time_s:.2f}s over {len(self.summary) - 1} steps"
        )

    def stationary_distribution(self) -> npt.NDArray[np.float64]:
        """
        Computes the analytic equilibrium (stationary) distribution of the Markov chain - i.e. the
        left eigenvector of the transition matrix corresponding to eigenvalue 1 - for comparison
        against the simulated results.
        """
        return self._stationary_distribution(self.transition_matrix)


class ConditionalMarkovChain(MarkovChainBase):
    """
    Simulates the same kind of population as MarkovChain, but applies a different transition matrix
    per row depending on group membership, via `no.df.transition_conditional`, instead of a single
    population-wide matrix. Run alongside a plain MarkovChain (same states/timeline/population size)
    to compare the resulting equilibria - see `mixed_stationary_distribution()` for why they can
    differ even though every individual's transitions follow the same rules either way.
    """

    def __init__(
        self,
        timeline: no.Timeline,
        npeople: int,
        states: npt.NDArray[np.int64],
        group_transition_matrices: dict[str, npt.NDArray[np.float64]],
    ) -> None:
        super().__init__(timeline, npeople, states)

        for group, matrix in group_transition_matrices.items():
            if not np.allclose(matrix.sum(axis=1), 1.0):
                raise ValueError(f"each row of the '{group}' transition matrix must sum to 1")
        self.group_transition_matrices = group_transition_matrices

        groups = list(group_transition_matrices)
        self.pop["group"] = pd.Categorical([groups[i % len(groups)] for i in range(npeople)], categories=groups)

    def step(self) -> None:
        self.pop["state"] = no.df.transition_conditional(
            self.mc, self.group_transition_matrices, self.pop["group"], self.pop["state"]
        )
        self.summary.loc[len(self.summary)] = self._state_counts()

    def mixed_stationary_distribution(self) -> npt.NDArray[np.float64]:
        """
        Computes the population-share-weighted average of each group's own analytic stationary
        distribution. Since group membership never changes, this is what the population actually
        converges to under transition_conditional whenever the groups' dynamics differ.
        """
        weights = self.pop["group"].value_counts(normalize=True)
        mixed = np.zeros(len(self.states))
        for group, matrix in self.group_transition_matrices.items():
            mixed += weights[group] * self._stationary_distribution(matrix)
        return mixed
