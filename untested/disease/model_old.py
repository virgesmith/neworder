
import pandas as pd
import numpy as np
import neworder

from matplotlib import pyplot as plt

from data import *

# possible transitions (1):
#        ----------------> 5
#       /    /    /    /
# 0 -> 1 -> 2 -> 3 -> 4 -> 6


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
    self.pop.loc[0, "tInfected"] = 0.0
    self.pop.loc[0, "tMild"] = neworder.mc.stopping(lambda_01, 1)

    self.pinfect = np.zeros(neworder.timeline.nsteps()+1)

    self.summary = pd.DataFrame(columns = ALLSTATES)

  def step(self):

    propcontagious = len(self.pop[self.pop.State.isin(INFECTIOUS)])/len(self.pop)
    pinfect = contagiousness * propcontagious
    self.pinfect[neworder.timeline.index] = pinfect

    # new infections
    self.pop.infected = self.pop.infected | neworder.mc.hazard(pinfect, self.npeople) #[bool(s) for s in neworder.mc.hazard(pinfect, self.npeople)]
    is_newly_infected = self.pop[(self.pop.infected) & (self.pop.State == State.UNINFECTED)]
    new_infections = len(is_newly_infected.index)
    neworder.log("new infections: %d" % new_infections)

    if new_infections > 0:

      # time of infection
      self.pop.loc[is_newly_infected.index, "tInfected"] = neworder.timeline.time()
      # current state
      self.pop.loc[is_newly_infected.index, "State"] = State.ASYMPTOMATIC
      # deal with disease progression in newly infected
      # # onset of (mild) symptoms
      # self.pop.loc[is_newly_infected.index, "tMild"] = self.pop.loc[is_newly_infected.index, "tInfected"] \
      #                                             + neworder.mc.stopping(lambda_12, new_infections)
      # # p(mild->severe)
      # h = neworder.mc.hazard(p_23, new_infections).astype(bool)
      # got_severe = is_newly_infected[h]
      # if len(got_severe) > self.npeople * beds_pct:
      #   neworder.log("EXCEEDED CARE CAPACITY")
      # recover = is_newly_infected[np.logical_not(h)]
      # self.pop.loc[got_severe.index, "tSevere"] = self.pop.loc[got_severe.index, "tMild"] \
      #                                           + neworder.mc.stopping(lambda_23, len(got_severe))
      # self.pop.loc[recover.index, "tRecovered"] = self.pop.loc[recover.index, "tMild"] \
      #                                           + neworder.mc.stopping(lambda_25, len(recover))

      # # p(severe->critical)
      # h = neworder.mc.hazard(p_34, len(got_severe.index)).astype(bool)
      # got_critical = got_severe[h]
      # if len(got_critical) > self.npeople * ccu_beds_pct:
      #   neworder.log("EXCEEDED CRITICAL CARE CAPACITY")
      # recover = got_severe[np.logical_not(h)]
      # self.pop.loc[got_critical.index, "tCritical"] = self.pop.loc[got_critical.index, "tSevere"] \
      #                                           + neworder.mc.stopping(lambda_34, len(got_critical))
      # self.pop.loc[recover.index, "tRecovered"] = self.pop.loc[recover.index, "tSevere"] \
      #                                         + neworder.mc.stopping(lambda_35, len(recover))

      # # p(critical->deceased)
      # h = neworder.mc.hazard(p_46, len(got_critical.index)).astype(bool)
      # die = got_critical[h]
      # recover = got_critical[np.logical_not(h)]
      # self.pop.loc[die.index, "tDeceased"] = self.pop.loc[die.index, "tCritical"] \
      #                                           + neworder.mc.stopping(lambda_46, len(die))
      # self.pop.loc[recover.index, "tRecovered"] = self.pop.loc[recover.index, "tCritical"] \
      #                                         + neworder.mc.stopping(lambda_45, len(recover))

      # update statuses
      self.pop.loc[self.pop.tMild < neworder.timeline.time(), "State"] = State.MILD
      self.pop.loc[self.pop.tSevere < neworder.timeline.time(), "State"] = State.SEVERE
      self.pop.loc[self.pop.tCritical < neworder.timeline.time(), "State"] = State.CRITICAL
      self.pop.loc[self.pop.tRecovered < neworder.timeline.time(), "State"] = State.RECOVERED
      self.pop.loc[self.pop.tDeceased < neworder.timeline.time(), "State"] = State.DECEASED

    self.summary = self.summary.append(self.pop.State.value_counts())

  def plot(self):
    self.summary = self.summary.fillna(0)
    self.summary.index = range(1,len(self.summary)+1)
    # force ordering for stacked bar chart
    self.summary = self.summary[[State.UNINFECTED, State.ASYMPTOMATIC, State.MILD, State.SEVERE, State.CRITICAL, State.RECOVERED, State.DECEASED]]
    neworder.log(self.summary)
    plt.plot(range(neworder.timeline.nsteps()+1), self.pinfect)
    #plt.plot(range(1,neworder.timeline.nsteps()+1), self.summary[State.DECEASED])

    self.summary.plot(kind='bar', width=1.0, stacked=True)

    #neworder.log("Overall mortality: %.2f%" % (self.summary.tail(1)[State.DECEASED].values[0] / self.npeople * 100.0))
    plt.show()



