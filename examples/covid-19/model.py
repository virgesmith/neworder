
import pandas as pd
import numpy as np
import neworder

from matplotlib import pyplot as plt

from data import *

# possible transitions:
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

    self.r = R0 ** (neworder.timeline.dt() / g) - 1.0 # per-timestep growth rate

    num_initial_infections = int(self.npeople * initial_infection_rate)
    patients_zero = range(0,num_initial_infections)

    # patient(s) zero & their outcomes
    self.pop.loc[patients_zero, "State"] = State.ASYMPTOMATIC
    self.pop.loc[patients_zero, "tInfected"] = 0.0
    # onset of (mild) symptoms, or recovery
    mild_disease, _ = self._transition(self.pop[0:num_initial_infections].index, "tInfected", [lambda_12, lambda_15], ["tMild", "tRecovered"])
    # onset of severe symptoms, or recovery
    severe_disease, _ = self._transition(mild_disease, "tMild", [lambda_23, lambda_25], ["tSevere", "tRecovered"])
    # onset of critical symptoms, or recovery - TODO should be dependent on availability of medical
    critical_disease, _ = self._transition(severe_disease, "tSevere", [lambda_34, lambda_35], ["tCritical", "tRecovered"])
    # onset of death or recovery
    self._transition(critical_disease, "tCritical", [lambda_46, lambda_45], ["tDeceased", "tRecovered"])

    self.infection_rate = np.zeros(neworder.timeline.nsteps()+1)
    self.mortality_rate = np.zeros(neworder.timeline.nsteps()+1)

    self.summary = pd.DataFrame(columns = ALLSTATES)

  def _transition(self, candidate_index, current_label, lambdas, labels):

    #TODO >2 possible states?
    #TODO C++ implementation?

    # compute arrival times for first transition
    t0 = neworder.mc.stopping(lambdas[0], len(candidate_index))
    self.pop.loc[candidate_index, labels[0]] = self.pop.loc[candidate_index, current_label] + t0
    # compute arrival times for second transition
    t1 = neworder.mc.stopping(lambdas[1], len(candidate_index))
    self.pop.loc[candidate_index, labels[1]] = self.pop.loc[candidate_index, current_label] + t1

    # what actually happens is what happens first
    # get indices for the 2 new states 
    i0 = candidate_index.intersection(self.pop[self.pop[labels[0]] < self.pop[labels[1]]].index)
    i1 = candidate_index.intersection(self.pop[self.pop[labels[0]] >= self.pop[labels[1]]].index)
    # remove arrival times for the event that wasn't first
    self.pop.loc[i1, labels[0]] = np.nan 
    self.pop.loc[i0, labels[1]] = np.nan

    return i0, i1


  def step(self):

    # expected rate of new infections
    raw_infection_rate = len(self.pop[self.pop.State.isin(INFECTIOUS)]) * self.r / self.npeople

    # new infections
    uninfected = self.pop.State == State.UNINFECTED
    h = neworder.mc.hazard(raw_infection_rate, self.npeople).astype(bool)
    newly_infected = self.pop[h & (self.pop.State == State.UNINFECTED)]
    #self.pop.loc[is_newly_infected.index, "State"] = State.ASYMPTOMATIC
    # self.pop.infected = self.pop.infected | neworder.mc.hazard(pinfect, self.npeople) #[bool(s) for s in neworder.mc.hazard(pinfect, self.npeople)]
    # is_newly_infected = self.pop[(self.pop.infected) & (self.pop.State == State.UNINFECTED)]
    new_infections = len(newly_infected.index)
    self.infection_rate[neworder.timeline.index()] = new_infections / sum(uninfected)
    neworder.log("eff infection rate %.2f%% new : %d" % (100.0 * new_infections / sum(uninfected), new_infections))

    if new_infections > 0:
      # time of infection
      self.pop.loc[newly_infected.index, "tInfected"] = neworder.timeline.time()
      
      # deal with disease progression in newly infected
      
      # onset of (mild) symptoms, or recovery
      mild_disease, _ = self._transition(newly_infected.index, "tInfected", [lambda_12, lambda_15], ["tMild", "tRecovered"])

      # onset of severe symptoms, or recovery
      severe_disease, _ = self._transition(mild_disease, "tMild", [lambda_23, lambda_25], ["tSevere", "tRecovered"])

      # onset of critical symptoms, or recovery - TODO should be dependent on availability of medical
      critical_disease, _ = self._transition(severe_disease, "tSevere", [lambda_34, lambda_35], ["tCritical", "tRecovered"])

      # onset of death or recovery
      self._transition(critical_disease, "tCritical", [lambda_46, lambda_45], ["tDeceased", "tRecovered"])

      # update statuses
      self.pop.loc[self.pop.tInfected < neworder.timeline.time(), "State"] = State.ASYMPTOMATIC
      self.pop.loc[self.pop.tMild < neworder.timeline.time(), "State"] = State.MILD
      self.pop.loc[self.pop.tSevere < neworder.timeline.time(), "State"] = State.SEVERE
      self.pop.loc[self.pop.tCritical < neworder.timeline.time(), "State"] = State.CRITICAL
      self.pop.loc[self.pop.tRecovered < neworder.timeline.time(), "State"] = State.RECOVERED
      self.pop.loc[self.pop.tDeceased < neworder.timeline.time(), "State"] = State.DECEASED
    
    self.summary = self.summary.append(self.pop.State.value_counts())

  def finalise(self):
 
    deaths = sum(~neworder.isnever(self.pop.tDeceased.values))
    # simple measure of test coverage
    observed_cases = sum(~neworder.isnever(self.pop.tSevere.values)) + 0.25 * sum(~neworder.isnever(self.pop.tMild.values))

    neworder.log("Mortality: observed = %.2f%%, actual = %.f%%" % (100.0 * deaths / observed_cases, 100.0 * deaths / self.npeople))
 
    # format the summary table
    self.summary = self.summary.fillna(0)
    self.summary.index = range(1,len(self.summary)+1)
    # use the string representations of the int enums
    self.summary.rename(columns={s: State(s).name for s in self.summary.columns.values}, inplace=True) 




