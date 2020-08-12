
import pandas as pd
import numpy as np
import neworder

from matplotlib import pyplot as plt

from data import *

# possible transitions:
# 0 -> 1
#  \
#   -> 2

class MarkovChain:
  def __init__(self, npeople):
    self.npeople = npeople
    self.cont = pd.DataFrame(data = {"State": [State.INIT] * npeople,  
                                    "t1": np.nan, 
                                    "t2": np.nan })

    self.disc = pd.DataFrame(data = {"State": [State.INIT] * npeople,  
                                    "t1": np.nan, 
                                    "t2": np.nan })

    self.cont.loc[0, "State"] = State.INIT

    self.csummary = pd.DataFrame(columns = ALLSTATES)
    self.dsummary = pd.DataFrame(columns = ALLSTATES)

    self.cont["t1"] = neworder.mc.stopping(lambda_01, len(self.cont))
    self.cont["t2"] = neworder.mc.stopping(lambda_02, len(self.cont))

    # obliterate t1s that occur before t2 and vice versa
    t2first = self.cont.loc[self.cont["t1"] > self.cont["t2"]].index
    t1first = self.cont.index.difference(t2first)
    self.cont.loc[t2first, "t1"] = np.nan
    self.cont.loc[t1first, "t2"] = np.nan

    # transition matrix for discrete model
    self.t = np.zeros((3,3))
    self.t[1,0] = lambda_01 * neworder.timeline.dt() #- 0.5 * (lambda_01 * neworder.timeline.dt())**2
    self.t[2,0] = lambda_02 * neworder.timeline.dt() #- 0.5 * (lambda_02 * neworder.timeline.dt())**2
    self.t[0,0] = 1.0 - self.t[1,0] - self.t[2,0]
    self.t[1,1] = 1.0
    self.t[2,2] = 1.0
    neworder.log(self.t)

  def step(self):
    # update continuous statuses
    self.cont.loc[self.cont.t1 < neworder.timeline.time(), "State"] = State.STATE1
    self.cont.loc[self.cont.t2 < neworder.timeline.time(), "State"] = State.STATE2
    self.csummary = self.csummary.append(self.cont.State.value_counts())

    # update discrete
    neworder.dataframe.transition(ALLSTATES, self.t, self.disc, "State")
    self.dsummary = self.dsummary.append(self.disc.State.value_counts())

  def plot(self):
    self.csummary = self.csummary.fillna(0)
    self.dsummary = self.dsummary.fillna(0)
    self.csummary.index = range(1,len(self.csummary)+1)
    self.dsummary.index = range(1,len(self.dsummary)+1)
    # force ordering for stacked bar chart
    # self.csummary = self.csummary[[State.INIT, State.STATE1, State.STATE2]]
    # self.dsummary = self.dsummary[[State.INIT, State.STATE1, State.STATE2]]

    plt.plot(range(1,neworder.timeline.nsteps()+1), np.log(self.csummary[State.INIT]))
    plt.plot(range(1,neworder.timeline.nsteps()+1), np.log(self.dsummary[State.INIT]))

    self.csummary.plot(kind='bar', width=1.0, stacked=True)
    self.dsummary.plot(kind='bar', width=1.0, stacked=True)

    neworder.log("Final Occupancy [C]:")
    neworder.log(self.csummary.tail(1))
    neworder.log("Final Occupancy [D]:")
    neworder.log(self.dsummary.tail(1))
    plt.show()



