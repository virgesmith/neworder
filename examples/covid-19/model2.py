
""" 2nd attempt at model """

# possible transitions (2):
#         ----------> 5
#       /    /    /
# 0 -> 1 -> 2 -> 3 <-
#                 \  \  
#                  -> 4 -> 6

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import neworder
from data import *

class Model:
  def __init__(self, npeople):
    self.npeople = npeople
    self.pop = pd.DataFrame(data = {"State": [State.UNINFECTED] * npeople,  
                                    "tInfected": np.nan, 
                                    "tMild": np.nan, 
                                    "tSevere": np.nan, 
                                    "tCritical": np.nan, 
                                    "tRecovered": np.nan,
                                    "tDeceased": np.nan })

    # patient zero
    self.pop.loc[0, "State"] = State.ASYMPTOMATIC
    self.pop.loc[0, "tInfected"] = 0.0 # neworder.timeline.time()
    self.pop.loc[0, "tMild"] = neworder.mc.stopping(lambda_01, 1)

    # probabilities that are a function of the overall state,
    self.p_infect = np.zeros(neworder.timeline.nsteps()+1)
    self.p_critical = np.zeros(neworder.timeline.nsteps()+1)
    self.p_die = np.zeros(neworder.timeline.nsteps()+1)
    
    self.summary = pd.DataFrame(columns = ALLSTATES)

    self.transitions = np.zeros((NUMSTATES,NUMSTATES))
    # from uninfected
    dt = 1.0
    propcontagious = len(self.pop[self.pop.State.isin(INFECTIOUS)])/len(self.pop)
    p01 = 10.0 * propcontagious   

    self.transitions[0,0] = 1 - p01 * dt
    self.transitions[1,0] = p01 * dt
    # from asymptomatic
    self.transitions[1,1] = 1 - p_12 * dt - p_15 * dt
    self.transitions[2,1] = p_12 * dt
    self.transitions[5,1] = p_15 * dt
    # from mild
    self.transitions[2,2] = 1 - p_23 * dt - p_25 * dt
    self.transitions[3,2] = p_23 * dt
    self.transitions[5,2] = p_25 * dt
    # from severe
    self.transitions[3,3] = 1 - p_34 * dt - p_35 * dt
    self.transitions[4,3] = p_34 * dt
    self.transitions[5,3] = p_35 * dt
    # from critical
    p_43 = 0.1
    self.transitions[3,4] = p_43 * dt
    self.transitions[4,4] = 1 - p_43 * dt - p_46 * dt
    self.transitions[6,4] = p_46 * dt
    # from recovered/dead
    self.transitions[5,5] = 1.0
    self.transitions[6,6] = 1.0

    #self.transitions = np.transpose(self.transitions)

  def step(self):
    # neworder.log(self.transitions)
    # neworder.log(self.transitions.sum(1))
    neworder.transition(ALLSTATES, self.transitions, self.pop, "State")
    self.summary = self.summary.append(self.pop.State.value_counts())

  def plot(self):
    self.summary = self.summary.fillna(0)
    self.summary.index = range(1,len(self.summary)+1)
    # force ordering for stacked bar chart
    #self.summary = self.summary[[State.UNINFECTED, State.ASYMPTOMATIC, State.MILD, State.SEVERE, State.CRITICAL, State.RECOVERED, State.DECEASED]]

    self.summary.plot(kind='bar', width=1.0, stacked=True)
    plt.show()
