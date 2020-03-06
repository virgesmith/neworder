
import pandas as pd
import numpy as np
from enum import Enum
import neworder

from matplotlib import pyplot as plt

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

contagiousness = 0.18
mild_symptoms_onset_tmean = 7.0 # days
p_get_severe = 0.2 # otherwise recover
mild_symptoms_recover_tmean = 7.0
severe_symptoms_onset_tmean = 7.0
p_recover_from_severe = 0.75 #
severe_symptoms_recover_tmean = 7.0
severe_symptoms_die_tmean = 7.0


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
    self.mild_onset_rate = 1.0 / mild_symptoms_onset_tmean
    self.severe_onset_rate = 1.0 # / severe_symptoms_onset_tmean
    self.recover_mild_rate = 1.0 / mild_symptoms_recover_tmean
    self.recover_severe_rate = 1.0 / severe_symptoms_recover_tmean
    self.die_severe_rate = 1.0 / severe_symptoms_die_tmean

    # patient zero
    self.pop.loc[0, "infected"] = True
    self.pop.loc[0, "State"] = State.ASYMPTOMATIC
    self.pop.loc[0, "tInfected"] = 0.0
    self.pop.loc[0, "tMild"] = neworder.mc.stopping(self.mild_onset_rate, 1)

    self.pinfect = np.zeros(neworder.timeline.nsteps()+1)

    self.summary = pd.DataFrame()

  def step(self):

    propcontagious = len(self.pop[self.pop.State.isin(INFECTIOUS)])/len(self.pop)
    pinfect = contagiousness * propcontagious
    self.pinfect[neworder.timeline.index()] = pinfect

    # new infections
    self.pop.infected = self.pop.infected | neworder.mc.hazard(pinfect, self.npeople) #[bool(s) for s in neworder.mc.hazard(pinfect, self.npeople)]
    is_newly_infected = self.pop[(self.pop.infected) & (self.pop.State == State.UNINFECTED)]
    new_infections = len(is_newly_infected.index)
    neworder.log("new infections: %d" % new_infections)

    if new_infections == 0:
      return
      
    # time of infection
    #self.pop.loc[(self.pop.infected) & (self.pop.State == State.UNINFECTED), "tInfected"] = neworder.timeline.time()
    self.pop.loc[is_newly_infected.index, "tInfected"] = neworder.timeline.time()
    # current state
    self.pop.loc[is_newly_infected.index, "State"] = State.ASYMPTOMATIC
    # deal with disease progression in newly infected
    # onset of (mild) symptoms
    self.pop.loc[is_newly_infected.index, "tMild"] = self.pop.loc[is_newly_infected.index, "tInfected"] \
                                                + neworder.mc.stopping(self.mild_onset_rate, new_infections)
    # p(mild->severe)
    h = neworder.mc.hazard(p_get_severe, new_infections).astype(bool)
    #neworder.log(h)
    #neworder.log(is_newly_infected)
    got_severe = is_newly_infected[h]
    recover = is_newly_infected[np.logical_not(h)]
    #neworder.log(recover)
    #neworder.log("severe infections: %d" % len(got_severe))
    #self.pop.loc[got_severe, "tSevere"] = self.pop.loc[got_severe, "tMild"] + neworder.mc.stopping(self.severe_onset_rate, len(got_severe))
    self.pop.loc[got_severe.index, "tSevere"] = self.pop.loc[got_severe.index, "tMild"] \
                                              + neworder.mc.stopping(self.severe_onset_rate, len(got_severe))
    #neworder.log("recoveries: %d" % len(recover))    
    self.pop.loc[recover.index, "tRecovered"] = self.pop.loc[recover.index, "tMild"] \
                                              + neworder.mc.stopping(self.recover_mild_rate, len(recover))

    # p(severe->recover)
    h = neworder.mc.hazard(p_recover_from_severe, len(got_severe.index)).astype(bool)
    recover = got_severe[h]
    die = got_severe[np.logical_not(h)]
    self.pop.loc[recover.index, "tRecovered"] = self.pop.loc[recover.index, "tSevere"] \
                                              + neworder.mc.stopping(self.recover_severe_rate, len(recover))
    self.pop.loc[die.index, "tDeceased"] = self.pop.loc[die.index, "tSevere"] \
                                             + neworder.mc.stopping(self.die_severe_rate, len(die))

    # update statuses
    self.pop.loc[self.pop.tMild < neworder.timeline.time(), "State"] = State.MILD
    self.pop.loc[self.pop.tSevere < neworder.timeline.time(), "State"] = State.SEVERE
    self.pop.loc[self.pop.tRecovered < neworder.timeline.time(), "State"] = State.RECOVERED
    self.pop.loc[self.pop.tDeceased < neworder.timeline.time(), "State"] = State.DECEASED
    
    self.summary = self.summary.append(self.pop.State.value_counts())

  def plot(self):
    self.summary = self.summary.fillna(0)
    self.summary.index = range(1,len(self.summary)+1)
    neworder.log(self.summary)
    #plt.plot(range(neworder.timeline.nsteps()+1), self.pinfect)

    # b = [ self.pop.tMild[~neworder.isnever(self.pop.tMild.values)], 
    #       self.pop.tSevere[~neworder.isnever(self.pop.tSevere.values)],
    #       self.pop.tRecovered[~neworder.isnever(self.pop.tRecovered.values)],
    #       self.pop.tDeceased[~neworder.isnever(self.pop.tDeceased.values)] ]

    # plt.hist(b, bins=range(neworder.timeline.nsteps()+1), stacked=True)
    self.summary.plot(kind='bar', width=1.0, stacked=True)

    plt.show()


