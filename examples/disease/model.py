
"""
Infectious disease case-based microsimulation model
Pathways are Poisson processes

possible state transitions:
        ----------> 5
      /    /    /
0 -> 1 -> 2 -> 3 <-
                \  \
                 -> 4 -> 6

(see data.py for explanation of values)
"""

import pandas as pd
import numpy as np

import neworder

from data import *
from graphics import Graphics

class DiseaseModel(neworder.Model):
  def __init__(self, days, dt, npeople):

    super().__init__(neworder.Timeline(0, days, [int(days/dt)]))

    n = self.timeline().nsteps()

    self.npeople = npeople
    self.pop = pd.DataFrame(data = {"State": [State.UNINFECTED] * npeople,
                                    "tInfected": np.nan,
                                    "tMild": np.nan,
                                    "tSevere": np.nan,
                                    "tCritical": np.nan,
                                    "tRecovered": np.nan,
                                    "tDeceased": np.nan })

    # probabilities that are a function of the overall state,
    self.p_infect = np.zeros(n+1)
    self.p_critical = np.zeros(n+1)
    self.p_die = np.zeros(n+1)

    self.summary = pd.DataFrame(columns = ALLSTATES)

    self.transitions = np.zeros((NUMSTATES,NUMSTATES))
    # from uninfected

    dt = self.timeline().dt()

    self.r = R0 ** (dt / g) - 1.0 # per-timestep growth rate

    num_initial_infections = int(self.npeople * initial_infection_rate)
    patients_zero = range(0,num_initial_infections)
    # patients zero
    self.pop.loc[patients_zero, "State"] = State.ASYMPTOMATIC
    self.pop.loc[patients_zero, "tInfected"] = 0.0 # self.timeline().time()

    self.infection_rate = np.zeros(n+1)
    self.mortality_rate = np.zeros(n+1)

    # from uninfected
    # from asymptomatic
    self.transitions[State.ASYMPTOMATIC,State.ASYMPTOMATIC] = 1 - lambda_12 * dt - lambda_15 * dt
    self.transitions[State.MILD,        State.ASYMPTOMATIC] = lambda_12 * dt
    self.transitions[State.RECOVERED,   State.ASYMPTOMATIC] = lambda_15 * dt
    # from mild
    self.transitions[State.MILD,        State.MILD]         = 1 - lambda_23 * dt - lambda_25 * dt
    self.transitions[State.SEVERE,      State.MILD]         = lambda_23 * dt
    self.transitions[State.RECOVERED,   State.MILD]         = lambda_25 * dt
    # from severe
    self.transitions[State.SEVERE,      State.SEVERE]       = 1 - lambda_34 * dt - lambda_35 * dt
    self.transitions[State.CRITICAL,    State.SEVERE]       = lambda_34 * dt
    self.transitions[State.RECOVERED,   State.SEVERE]       = lambda_35 * dt
    # from critical
    # TODO back to severe?
    self.transitions[State.CRITICAL,    State.CRITICAL]     = 1 - lambda_45 * dt - lambda_46 * dt
    self.transitions[State.RECOVERED,   State.CRITICAL]     = lambda_45 * dt
    self.transitions[State.DECEASED,    State.CRITICAL]     = lambda_46 * dt
    # from recovered/dead
    self.transitions[State.RECOVERED,   State.RECOVERED]    = 1.0
    self.transitions[State.DECEASED,    State.DECEASED]     = 1.0

    self.severe_care_cap = self.npeople * beds_pct
    self.critical_care_cap = self.npeople * ccu_beds_pct

    #self.transitions = np.transpose(self.transitions)
    #neworder.log(self.transitions)

  def transition(self):
    self.step()

  def checkpoint(self):
    self.finalise()
    Graphics().plot(self)

  def _update_t(self, previous_states, new_state, new_state_label):
    index = self.pop[(previous_states != new_state) & (self.pop.State == new_state)].index
    self.pop.loc[index, new_state_label] = self.timeline().time()
    return index

  def step(self):

    # implement social distancing
    if self.timeline().time() == social_distancing_policy[0]:
      self.r = self.r * social_distancing_policy[1]
      neworder.log("Social distancing implemented: r changed to %.2f%%" % (self.r * 100.0))

    previous_state = self.pop.State.copy()

    if len(self.pop[self.pop.State == State.SEVERE]) > self.severe_care_cap:
      neworder.log("Non-CCU capcity exceeded")
    if len(self.pop[self.pop.State == State.CRITICAL]) > self.critical_care_cap:
      neworder.log("CCU capcity exceeded")

    self.severe_care_cap = self.npeople * beds_pct
    self.critical_care_cap = self.npeople * ccu_beds_pct

    dt = self.timeline().dt()

    # adjust infection transition
    raw_infection_rate = len(self.pop[self.pop.State.isin(INFECTIOUS)]) * self.r / self.npeople
    self.p_infect[self.timeline().index()] = raw_infection_rate
    self.transitions[State.UNINFECTED,State.UNINFECTED] = 1 - raw_infection_rate * dt
    self.transitions[State.ASYMPTOMATIC, State.UNINFECTED] = raw_infection_rate * dt

    # adjust severe->recovery transition according to bed capacity - make recovery less likely
    severe_adj = max(1.0, len(self.pop[self.pop.State == State.SEVERE]) / self.severe_care_cap)
    self.transitions[State.SEVERE,      State.SEVERE]       = 1 - lambda_34 * dt - lambda_35 * dt / severe_adj
    self.transitions[State.CRITICAL,    State.SEVERE]       = lambda_34 * dt
    self.transitions[State.RECOVERED,   State.SEVERE]       = lambda_35 * dt / severe_adj

    # adjust critical->recovery transition according to ccu bed capacity - make recovery less likely
    critical_adj = max(1.0, len(self.pop[self.pop.State == State.CRITICAL]) / self.critical_care_cap)
    self.transitions[State.DECEASED,   State.CRITICAL]      = lambda_46 * dt
    self.transitions[State.CRITICAL,    State.CRITICAL]     = 1 - lambda_45 * dt / critical_adj - lambda_46 * dt
    self.transitions[State.RECOVERED,    State.CRITICAL]    = lambda_45 * dt / critical_adj

    neworder.dataframe.transition(self, ALLSTATES, self.transitions, self.pop, "State")
    self.summary = self.summary.append(self.pop.State.value_counts())

    uninfected = self.pop[(previous_state == State.UNINFECTED)].index
    new_infections = self._update_t(previous_state, State.ASYMPTOMATIC, "tInfected")
    self._update_t(previous_state, State.MILD, "tMild")
    self._update_t(previous_state, State.SEVERE, "tSevere")
    self._update_t(previous_state, State.CRITICAL, "tCritical")
    new_deaths = self._update_t(previous_state, State.DECEASED, "tDeceased")

    # infection rate amongst those previously uninfected
    neworder.log("effective infection rate %.2f%% new : %d" % (100.0 * len(new_infections) / len(uninfected), len(new_infections)))
    self.infection_rate[self.timeline().index()] = len(new_infections) / len(uninfected)
    self.mortality_rate[self.timeline().index()] = len(new_deaths) / len(self.pop[self.pop.State != State.DECEASED])

  def finalise(self):

    deaths = sum(~neworder.time.isnever(self.pop.tDeceased.values))
    # simple measure of test coverage 100% or severe and above, 25% of mild
    observed_cases = sum(~neworder.time.isnever(self.pop.tSevere.values)) + 0.25 * sum(~neworder.time.isnever(self.pop.tMild.values))

    neworder.log("Mortality: observed = %.2f%%, actual = %.f%%" % (100.0 * deaths / observed_cases, 100.0 * deaths / self.npeople))

    self.summary = self.summary.fillna(0)
    self.summary.index = range(1,len(self.summary)+1)
    # use the string representations of thobserved_casese int enums
    self.summary.rename(columns={s: State(s).name for s in self.summary.columns.values}, inplace=True)



# initialise the model
days = 180
dt = 1
npeople = 10000

disease_model = DiseaseModel(days, dt, npeople)

neworder.run(disease_model)