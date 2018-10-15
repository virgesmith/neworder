""" RiskPaths model """

from enum import Enum
import pandas as pd
import neworder
from helpers import *
from matplotlib import pyplot as plt

# dynamics data
import data

AgeintState = partition(15, 40, 2.5)

TIME_INFINITE = -1.0

# classification UNION_STATE
class UnionState(Enum):
  NEVER_IN_UNION = 0
  FIRST_UNION_PERIOD1 = 1
  FIRST_UNION_PERIOD2 = 2
  AFTER_FIRST_UNION = 3
  SECOND_UNION = 4
  AFTER_SECOND_UNION = 5

def in_union(state):
  return state == FIRST_UNION_PERIOD1 or state == FIRST_UNION_PERIOD2 or state == SECOND_UNION

class Parity(Enum):
  CHILDLESS = 0
  PREGNANT = 1

class Union(Enum):
  FIRST = 0
  SECOND = 1

UnionDuration = [1, 3, 5, 9, 13]

# # TODO this would be so much more efficient as a struct-of-arrays rather than an array-of-structs
# class Person():
#   """ actor Person """
#   def __init__(self):
#     """ equivalent to Person::Start() """
#     self.age = 0
#     self.time = 0
#     self.alive = True # using boolean as opposed to LIFE_STATE classification
#     self.parity = Parity.CHILDLESS
#     #self.time_of_first_pregnancy = neworder.stopping(neworder.mortality_rate, 1)[0]
#     self.unions = 0
#     self.union_status = Union.NEVER_IN_UNION
#     self.union_period2_change = TIME_INFINITE

#     #print(self.time_of_death)

#   def __del__(self):
#     """ equivalent to Person::Finish() """
#     pass

#   def age_int(self):
#     """ rough equivalent of self-scheduling int age """
#     return int(self.age)

#   def age_status(self):
#     # interpolate 
#     pass

#   def status(self, t):
#     return self.time_death() > t

#   # Events
#   def time_death(self):
#     p = neworder.mortality_rate
#     if p >= 1.0:
#       return 0.0
#     else:
#       return min(neworder.stopping(p, 1)[0], neworder.timestep())

#   def death(self):
#     """ equivalent of Person::DeathEvent() """
#     self.alive = False

#   def time_first_pregnancy():
#     """ Equivalent to Person::timeFirstPregEvent() """
#     t = TIME_INFINITE
#     if self.parity == Parity.CHILDLESS:
#       p = AgeBaselinePreg1[self.age_status()] * UnionStatusPreg1[self.union_status()]
#       t = neworder.stopping(p, 1)[0]
#     return t

#   def first_pregnancy(self):
#     """ Equivalent of Person::FirstPregEvent() """
#     self.parity = Parity.PREGNANT

#   def union1_formation(self):
#     """ Equivalent of Person::Union1FormationEvent() """
#     self.unions = self.unions + 1
#     self.union_status = Union.FIRST
#     self.union_period2_change = 3.0

#   def time_union_period2(self):
#     """ Person::timeUnionPeriod2Event() """
#     return self.union_period2_change
  
#   def union_period2(self):
#     """ Person::UnionPeriod2Event() """
#     if self.union_status == UnionState.FIRST_UNION_PERIOD1:
#       self.union_status = UnionState.FIRST_UNION_PERIOD2
#     self.union_period2_change = TIME_INFINITE


class RiskPaths():
  def __init__(self, n):

    # TODO fertility and mortality

    # initialise population - time of death only 
    self.population = pd.DataFrame(data={"TimeOfDeath": neworder.first_arrival(data.mortality_rate, neworder.timestep, n, 0.0),
                                         "TimeOfPregnancy": np.full(n, neworder.never()),
                                         "Parity": np.full(n, Parity.CHILDLESS),
                                         "Unions": np.zeros(n, dtype=int),
                                         #"UnionStatus": np.full(n, UnionState.NEVER_IN_UNION),
                                         #"UnionPeriod2Change": np.full(n, TIME_INFINITE)
                                        })

    # Construct a timeline of unions for each person

    # minimum age for marriage is 15 (this was soviet-era data)
    # first union - probabilities start at 15, so we add this on afterwards
    self.population["T_Union1Start"] = neworder.first_arrival(data.p_u1f, data.delta_t, len(self.population)) + data.min_age
    self.population["T_Union1End"] = neworder.next_arrival(self.population["T_Union1Start"].values, data.r_diss2[0], data.delta_t_u, True, data.min_u1) 

    # second union
    self.population["T_Union2Start"] = neworder.next_arrival(self.population["T_Union1End"].values, data.r_u2f, data.delta_t, True) 
    # no mimimum time of 2nd union (?)
    self.population["T_Union2End"] = neworder.next_arrival(self.population["T_Union2Start"].values, data.r_diss2[1], data.delta_t_u, True) 

    # and discard events happening after death
    self.population.loc[self.population["T_Union1Start"] > self.population["TimeOfDeath"], "T_Union1Start"] = neworder.never()
    self.population.loc[self.population["T_Union1End"] > self.population["TimeOfDeath"], "T_Union1End"] = neworder.never()
    self.population.loc[self.population["T_Union2Start"] > self.population["TimeOfDeath"], "T_Union2Start"] = neworder.never()
    self.population.loc[self.population["T_Union2End"] > self.population["TimeOfDeath"], "T_Union2End"] = neworder.never()

    # count unions entered into
    self.population.Unions = (~neworder.isnever(self.population["T_Union1Start"].values)).astype(int) \
                           + (~neworder.isnever(self.population["T_Union2Start"].values)).astype(int)

    neworder.log("RiskPaths init")
  
  def plot(self):
    # plt.hist(self.population.TimeOfDeath, range(100), color='black')
    # b = [ self.population.T_Union1Start[~np.isnan(self.population.T_Union1Start)], 
    #       self.population.T_Union1End[~np.isnan(self.population.T_Union1End)],
    #       self.population.T_Union2Start[~np.isnan(self.population.T_Union2Start)],
    #       self.population.T_Union2End[~np.isnan(self.population.T_Union2End)] ]
    # plt.hist(b, range(100), stacked=True)
    #plt.savefig("./doc/examples/img/competing_hist_100k.png")
    #plt.show()
    #neworder.log(self.population)
    pass

  def stats(self):
    neworder.log("mean unions = %f" % np.mean(self.population.Unions))
    
  def pregnancy(self):
    # pre-union1 pregnancy
    p_preg = data.p_preg * data.r_preg[0] #UnionState.NEVER_IN_UNION]
    self.population["TimeOfPregnancy"] = neworder.first_arrival(p_preg, data.delta_t, len(self.population)) + data.min_age
    self.population.loc[self.population["TimeOfPregnancy"] > self.population["T_Union1Start"], "TimeOfPregnancy"] = neworder.never()
    #self.population.loc[self.population["TimeOfPregnancy"] > self.population["TimeOfDeath"], "TimeOfPregnancy"] = neworder.never()                    
    #self.population.loc[~neworder.isnever(self.population["TimeOfPregnancy"].values), "Parity"] = Parity.PREGNANT

    # union1 phase1 pregnancy
    p_preg = data.p_preg * data.r_preg[1] #UnionState.FIRST_UNION_PERIOD1]
    # # NaNs break this - minimum is always first
    self.population["TimeOfPregnancyU1a"] = neworder.next_arrival(self.population["T_Union1Start"].values, p_preg, data.delta_t)
    self.population.loc[self.population["TimeOfPregnancyU1a"] > self.population["T_Union1Start"] + data.min_u1, "TimeOfPregnancyU1a"] = neworder.never()                    

    # union1 phase2 pregnancy
    p_preg = data.p_preg * data.r_preg[2] #UnionState.FIRST_UNION_PERIOD2]
    self.population["TimeOfPregnancyU1b"] = neworder.next_arrival(self.population["T_Union1Start"].values + data.min_u1, p_preg, data.delta_t)
    self.population.loc[self.population["TimeOfPregnancyU1b"] > self.population["T_Union1End"], "TimeOfPregnancyU1b"] = neworder.never()                    

    # # post union1 pregnancy
    p_preg = data.p_preg * data.r_preg[3] #UnionState.AFTER_FIRST_UNION]
    print(neworder.next_arrival(self.population["T_Union1End"].values, p_preg, data.delta_t))
    print(neworder.next_arrival(self.population["T_Union1End"].values, p_preg, data.delta_t))
    self.population["TimeOfPregnancyPostU1"] = neworder.next_arrival(self.population["T_Union1End"].values, p_preg, data.delta_t)
    # #self.population.loc[self.population["TimeOfPregnancyPostU1"] > self.population["T_Union2Start"], "TimeOfPregnancyPostU1"] = neworder.never()                    

    # # union2 pregnancy
    # p_preg = data.p_preg * data.r_preg[4] #UnionState.SECOND_UNION]
    # self.population["TimeOfPregnancyU2"] = neworder.next_arrival(self.population["T_Union2Start"].values, p_preg, data.delta_t)
    # #self.population.loc[self.population["TimeOfPregnancyU2"] > self.population["T_Union2Start"], "TimeOfPregnancyU2"] = neworder.never()                    

    # # post union2 pregnancy
    # p_preg = data.p_preg * data.r_preg[5] #UnionState.AFTER_SECOND_UNION]
    # self.population["TimeOfPregnancyPostU2"] = neworder.next_arrival(self.population["T_Union2End"].values, p_preg, data.delta_t)

    self.population.to_csv("./population.csv", index=False)

  def check(self):
    #assert len(self.population[self.population["TimeOfPregnancy"] > self.population["TimeOfDeath"]])
    return True
