import pandas as pd  # type: ignore
import numpy as np
import neworder as no


class MarkovChain(no.Model):
  def __init__(self, timeline: no.Timeline, npeople: int, states: np.ndarray, transition_matrix: np.ndarray) -> None:

    super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)
    self.npeople = npeople

    self.pop = pd.DataFrame(data={"state": np.full(npeople, 0),
                                  "t1": no.time.NEVER,
                                  "t2": no.time.NEVER})

    self.states = states
    self.transition_matrix = transition_matrix
    self.summary = pd.DataFrame(columns=states)
    self.summary.loc[0] = self.pop.state.value_counts().transpose()

  # pure python equivalent implementation of no.df.transition, to illustrate the performance gain
  def transition_py(self, colname: str) -> None:
    def _interp(cumprob: np.ndarray, x: float) -> int:
      lbound = 0
      while lbound < len(cumprob) - 1:
        if cumprob[lbound] > x:
          break
        lbound += 1
      return lbound

    def _sample(u: float, tc: np.ndarray, c: np.ndarray) -> float:
      return c[_interp(tc, u)]

    # u = m.mc.ustream(len(df))
    tc = np.cumsum(self.transition_matrix, axis=1)

    # reverse mapping of category label to index
    lookup = {self.states[i]: i for i in range(len(self.states))}

    # for i in range(len(df)):
    #   current = df.loc[i, colname]
    #   df.loc[i, colname] = sample(u[i], tc[lookup[current]], c)
    # this is a much faster equivalent of the loop in the commented code immediately above
    self.pop[colname] = self.pop[colname].apply(lambda current: _sample(self.mc.ustream(1), tc[lookup[current]], self.states))

  def step(self) -> None:
    # self.transition_py("state")
    # comment the above line and uncomment this line to use the faster C++ implementation
    no.df.transition(self, self.states, self.transition_matrix, self.pop, "state")
    self.summary.loc[len(self.summary)] = self.pop.state.value_counts().transpose()

  def finalise(self) -> None:
    self.summary["t"] = np.linspace(self.timeline.start, self.timeline.end, self.timeline.nsteps + 1)
    self.summary.reset_index(drop=True, inplace=True)
    self.summary.fillna(0, inplace=True)


