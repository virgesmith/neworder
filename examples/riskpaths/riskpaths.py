""" RiskPaths model """

from enum import Enum

import neworder

# classification UNION_STATE
class UnionState(Enum):
  NEVER_IN_UNION = 0
  FIRST_UNION_PERIOD1 = 1
  FIRST_UNION_PERIOD2 = 2
  AFTER_FIRST_UNION = 3
  SECOND_UNION = 4
  AFTER_SECOND_UNION = 5

class Parity(Enum):
  CHILDLESS = 0
  PREGNANT = 1

class Union(Enum):
  FIRST = 0
  SECOND = 1

# TODO this would be so much more efficient as a struct-of-arrays rather than an array-of-structs
class Person():
  """ actor Person """
  def __init__(self):
    """ equivalent to Person::Start() """
    self.age = 0
    self.time = 0
    self.alive = True # using boolean as opposed to LIFE_STATE classification
    self.time_of_death = min(neworder.stopping(neworder.mortality_rate, 1)[0], neworder.timespan[1])
    self.parity = Parity.CHILDLESS
    self.time_of_first_pregnancy = neworder.stopping(neworder.mortality_rate, 1)[0]
    self.unions = 0
    self.time_of_union = [0.0, 0.0] # TODO sample this

    print(self.time_of_death)

  def __del__(self):
    """ equivalent to Person::Finish() """
    pass

  def status(self, t):
    return self.time_of_death > t

  # Events

  def death(self):
    """ equivalent of Person::DeathEvent() """
    self.alive = False

  def first_pregnancy(self):
    """ Equivalent of Person::FirstPregEvent() """
    self.parity = Parity.PREGNANT

  def union1_formation(self):
    """ Equivalent of Person::Union1FormationEvent() """
    self.unions = self.unions + 1
    self.union_status = Union.FIRST
    # union_period2_change = WAIT(3)

class RiskPaths():
  def __init__(self, n):
    # TODO seed
    # TODO check this affects C++ functions
    self.ustream = neworder.UStream(0)

    self.population = [ Person() for _ in range(n) ]
    
  def alive(self):
    # refactor
    ret = 0.0
    for i in range(len(self.population)):
      ret = ret + self.population[i].status(neworder.time)
    neworder.log("{} {:.2f}%".format(neworder.time, 100.0 * ret / len(self.population)))

