import pandas as pd
import numpy as np
import neworder as no

class MarkovChain(no.Model):
  def __init__(self, timeline, npeople, states, transition_matrix):

    super().__init__(timeline, no.MonteCarlo.deterministic_identical_stream)
    self.npeople = npeople

    self.pop = pd.DataFrame(data = {"state": np.full(npeople, 0),
                                    "t1": no.time.never(),
                                    "t2": no.time.never() })

    self.states = states
    self.transition_matrix = transition_matrix
    self.summary = pd.DataFrame(columns = states)
    self.summary = self.summary.append(self.pop.state.value_counts().transpose())

  # pure python equivalent implementation of neworder.dataframe.transition, to illustrate the performance gain
  def transition_py(self, colname):
    def _interp(cumprob, x):
      lbound = 0
      while lbound < len(cumprob) - 1:
        if cumprob[lbound] > x:
          break
        lbound += 1
      return lbound

    def _sample(u, tc, c):
      return c[_interp(tc, u)]

    #u = m.mc().ustream(len(df))
    tc = np.cumsum(self.transition_matrix, axis=1)

    # reverse mapping of category label to index
    lookup = { self.states[i]: i for i in range(len(self.states)) }

    # for i in range(len(df)):
    #   current = df.loc[i, colname]
    #   df.loc[i, colname] = sample(u[i], tc[lookup[current]], c)
    # this is a much faster equivalent of the loop in the commented code immediately above
    self.pop[colname] = self.pop[colname].apply(lambda current: _sample(self.mc().ustream(1), tc[lookup[current]], self.states))

  def step(self):
    #self.transition_py("state")
    # comment the above line and uncomment this line to use the faster C++ implementation
    no.dataframe.transition(self, self.states, self.transition_matrix, self.pop, "state")
    self.summary = self.summary.append(self.pop.state.value_counts().transpose())#, ignore_index=True)

  def checkpoint(self):
    self.summary["t"] = np.linspace(self.timeline().start(), self.timeline().end(), self.timeline().nsteps() + 1)
    #self.summary.set_index(np.linspace(self.timeline().start(), self.timeline().end(), self.timeline().nsteps() + 1), drop=True, inplace=True)
    self.summary.reset_index(drop=True, inplace=True)
    self.summary.fillna(0, inplace=True)


