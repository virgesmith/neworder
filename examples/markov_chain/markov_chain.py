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
    #self.cont.loc[0, "State"] = State.INIT

    #self.csummary = pd.DataFrame(columns = ALLSTATES)
    self.summary = pd.DataFrame(columns = states)
    self.summary = self.summary.append(self.pop.state.value_counts())

    # self.cont["t1"] = neworder.mc.stopping(lambda_01, len(self.cont))
    # self.cont["t2"] = neworder.mc.stopping(lambda_02, len(self.cont))

    # obliterate t1s that occur before t2 and vice versa
    # t2first = self.cont.loc[self.cont["t1"] > self.cont["t2"]].index
    # t1first = self.cont.index.difference(t2first)
    # self.cont.loc[t2first, "t1"] = np.nan
    # self.cont.loc[t1first, "t2"] = np.nan

  def step(self):
    no.dataframe.transition(self, self.states, self.transition_matrix, self.pop, "state")
    self.summary = self.summary.append(self.pop.state.value_counts(), ignore_index=True)
  
  def checkpoint(self):
    #self.summary["t"] = np.linspace(self.timeline().start(), self.timeline().end(), self.timeline().nsteps() + 1)
    self.summary.set_index(np.linspace(self.timeline().start(), self.timeline().end(), self.timeline().nsteps() + 1), drop=True, inplace=True)
    #self.summary.reset_index(drop=True, inplace=True)
    no.log(self.summary)


