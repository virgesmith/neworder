
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

    # probabilities that are a function of the overall state,
    self.p_infect = np.zeros(neworder.timeline.nsteps()+1)
    self.p_critical = np.zeros(neworder.timeline.nsteps()+1)
    self.p_die = np.zeros(neworder.timeline.nsteps()+1)
    
    self.summary = pd.DataFrame(columns = ALLSTATES)

    self.transitions = np.zeros((NUMSTATES,NUMSTATES))
    # from uninfected

    dt = neworder.timeline.dt()

    self.r = R0 ** (dt / g) - 1.0 # per-timestep growth rate

    num_initial_infections = int(self.npeople * initial_infection_rate)
    patients_zero = range(0,num_initial_infections)
    # patients zero
    self.pop.loc[patients_zero, "State"] = State.ASYMPTOMATIC
    self.pop.loc[patients_zero, "tInfected"] = 0.0 # neworder.timeline.time()

    # self.transitions[0,0] = 1 - p01 * dt
    # self.transitions[1,0] = p01 * dt
    #self.transitions[1,1] = 1.0
    self.transitions[2,2] = 1.0
    self.transitions[3,3] = 1.0
    self.transitions[4,4] = 1.0

    # from asymptomatic
    self.transitions[State.ASYMPTOMATIC, State.ASYMPTOMATIC] = 1 - lambda_12 * dt - lambda_15 * dt
    self.transitions[2,1] = lambda_12 * dt
    self.transitions[5,1] = lambda_15 * dt
    # from mild
    self.transitions[2,2] = 1 - lambda_23 * dt - lambda_25 * dt
    self.transitions[3,2] = lambda_23 * dt
    self.transitions[5,2] = lambda_25 * dt
    # from severe   
    self.transitions[3,3] = 1 - lambda_34 * dt - lambda_35 * dt
    self.transitions[4,3] = lambda_34 * dt
    self.transitions[5,3] = lambda_35 * dt
    # from critical 
    # TODO
    self.transitions[5,4] = lambda_46 * dt
    self.transitions[4,4] = 1 - lambda_45 * dt - lambda_46 * dt
    self.transitions[6,4] = lambda_45 * dt
    # from recovered/dead
    self.transitions[5,5] = 1.0
    self.transitions[6,6] = 1.0

    #neworder.log(self.transitions)
    #self.transitions = np.transpose(self.transitions)

  def step(self):
    raw_infection_rate = len(self.pop[self.pop.State.isin(INFECTIOUS)]) * self.r / self.npeople

    self.p_infect[neworder.timeline.index()] = raw_infection_rate
    self.transitions[0,0] = 1 - raw_infection_rate * neworder.timeline.dt()
    self.transitions[1,0] = raw_infection_rate * neworder.timeline.dt()

    neworder.transition(ALLSTATES, self.transitions, self.pop, "State")
    self.summary = self.summary.append(self.pop.State.value_counts())

  def plot(self):
    self.summary = self.summary.fillna(0)
    self.summary.index = range(1,len(self.summary)+1)
    # force ordering for stacked bar chart
    #self.summary = self.summary[[State.UNINFECTED, State.ASYMPTOMATIC, State.MILD, State.SEVERE, State.CRITICAL, State.RECOVERED, State.DECEASED]]

    neworder.log(self.summary)

    self.summary.plot(kind='bar', width=1.0, stacked=True)
    plt.show()
