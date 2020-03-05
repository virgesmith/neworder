
import pandas as pd
import numpy as np
from enum import Enum
import neworder

class State(Enum):
  UNINFECTED = 0
  ASYMPTOMATIC = 1
  MILD = 2
  SEVERE = 3
  RECOVERED = 4
  DECEASED = 5

INFECTIOUS = [State.ASYMPTOMATIC, State.MILD, State.SEVERE]



# possible transitions:
#             -> 4
#            /   ^
# 0 -> 1 -> 2 -> 3 -> 5

contagiousness = 0.002
mild_symptoms_rate = 0.02 # convert from t to p (-ln?)

class Model:
  def __init__(self, npeople):
    self.npeople = npeople
    self.pop = pd.DataFrame(data = {"infected": False,
                                    "State": [State.UNINFECTED] * npeople,  
                                    "tInfected": np.nan, 
                                    "tMild": np.nan, 
                                    "tSevere": np.nan, 
                                    "tRecovered": np.nan,
                                    "tDeceased": np.nan })

    # patient zero
    self.pop.loc[0, "infected"] = True
    self.pop.loc[0, "State"] = State.ASYMPTOMATIC
    self.pop.loc[0, "tInfected"] = 0.0
    self.pop.loc[0, "tMild"] = neworder.mc.stopping(mild_symptoms_rate, 1)


  def step(self):

    ncontagious = len(self.pop[self.pop.State.isin(INFECTIOUS)])
    pinfect = 1 - (1-contagiousness) ** ncontagious

    # new infections
    self.pop.infected = self.pop.infected | neworder.mc.hazard(pinfect, self.npeople) #[bool(s) for s in neworder.mc.hazard(pinfect, self.npeople)]
    newly_infected = self.pop[(self.pop.infected) & (self.pop.State == State.UNINFECTED)].index
    new_infections = len(newly_infected)
    neworder.log("new infections: %d" % new_infections)
    # time of infection
    self.pop.loc[(self.pop.infected) & (self.pop.State == State.UNINFECTED), "tInfected"] = neworder.timeline.time()
    # current state
    self.pop.loc[(self.pop.infected) & (self.pop.State == State.UNINFECTED), "State"] = State.ASYMPTOMATIC
    # disease progression
    # onset of (mild) symptoms
    self.pop.loc[newly_infected, "tMild"] = self.pop.loc[newly_infected, "tInfected"] + neworder.mc.stopping(mild_symptoms_rate, new_infections)

    # update statuses
    self.pop.loc[self.pop.tMild < neworder.timeline.time(), "State"] = State.MILD
    self.pop.loc[self.pop.tSevere < neworder.timeline.time(), "State"] = State.SEVERE
    self.pop.loc[self.pop.tRecovered < neworder.timeline.time(), "State"] = State.RECOVERED
    self.pop.loc[self.pop.tDeceased < neworder.timeline.time(), "State"] = State.DECEASED

    #neworder.log(self.pop) #[self.pop.State != State.UNINFECTED])
    self.pop.PrevState= self.pop.State
