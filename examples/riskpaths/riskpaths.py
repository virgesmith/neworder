""" RiskPaths model """

from enum import Enum

import neworder
from helpers import *

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

# TODO this would be so much more efficient as a struct-of-arrays rather than an array-of-structs
class Person():
  """ actor Person """
  def __init__(self):
    """ equivalent to Person::Start() """
    self.age = 0
    self.time = 0
    self.alive = True # using boolean as opposed to LIFE_STATE classification
    self.parity = Parity.CHILDLESS
    #self.time_of_first_pregnancy = neworder.stopping(neworder.mortality_rate, 1)[0]
    self.unions = 0
    self.union_period2_change = TIME_INFINITE

    #print(self.time_of_death)

  def __del__(self):
    """ equivalent to Person::Finish() """
    pass

  def age_int(self):
    """ rough equivalent of self-scheduling int age """
    return int(self.age)

  def age_status(self):
    # interpolate 
    pass

  def status(self, t):
    return self.time_death() > t

  # Events
  def time_death(self):
    p = neworder.mortality_rate
    if p >= 1.0:
      return 0.0
    else:
      return min(neworder.stopping(p, 1)[0], neworder.timespan[1])

  def death(self):
    """ equivalent of Person::DeathEvent() """
    self.alive = False

  def time_first_pregnancy():
    """ Equivalent to Person::timeFirstPregEvent() """
    t = TIME_INFINITE
    if self.parity == Parity.CHILDLESS:
      p = AgeBaselinePreg1[self.age_status()] * UnionStatusPreg1[self.union_status()]
      t = neworder.stopping(p, 1)[0]
    return t

  def first_pregnancy(self):
    """ Equivalent of Person::FirstPregEvent() """
    self.parity = Parity.PREGNANT

  def union1_formation(self):
    """ Equivalent of Person::Union1FormationEvent() """
    self.unions = self.unions + 1
    self.union_status = Union.FIRST
    self.union_period2_change = 3.0

  def time_union_period2(self):
    """ Person::timeUnionPeriod2Event() """
    return self.union_period2_change
  
  def union_period2(self):
    """ Person::UnionPeriod2Event() """
    if self.union_status == UnionState.FIRST_UNION_PERIOD1:
      self.union_status = UnionState.FIRST_UNION_PERIOD2
    self.union_period2_change = TIME_INFINITE


class RiskPaths():
  def __init__(self, n):
    # TODO seed
    # TODO check this affects C++ functions
    self.ustream = neworder.UStream(0)

    # initialise population
    self.population = [ Person() for _ in range(n) ]
    
  def alive(self):
    # refactor
    ret = 0.0
    for i in range(len(self.population)):
      ret = ret + self.population[i].status(neworder.time)
    neworder.log("{} {:.2f}%".format(neworder.time, 100.0 * ret / len(self.population)))


