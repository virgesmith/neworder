
import pandas as pd

import neworder
import ethpop

class Person():

  def __init__(self, mortality_hazard_file):
    """ Person::Start() """
    self.alive = True
    self.age = 0.0
    #self.time = 0.0
    self.mortality_hazard = ethpop.create(pd.read_csv(mortality_hazard_file), "E09000030").reset_index()

    self.mortality_hazard = self.mortality_hazard[(self.mortality_hazard.NewEthpop_ETH=="WBI") 
                                                & (self.mortality_hazard.DC1117EW_C_SEX==1)]

    self.time_mortality = neworder.TIME_INFINITY

  def finish(self):
    """ Person::Finish() """
    pass

  def state(self, t):
    """ Returns the person's state (alive/dead) at age t """
    return self.alive

  def inc_age(self):
    self.time_mortality_event()
    if self.alive:
      self.age = self.age + neworder.timestep

  def time_mortality_event(self):
    """ TIME Person::timeMortalityEvent() """
    t = neworder.stopping(self.mortality_hazard.Rate.values[min(neworder.timeindex-1, 85)], 1)[0]
    if t < neworder.timestep or self.age >= neworder.timespan[-2]:
      self.mortality_event(t)
    #neworder.log("TOD=%f" % self.time_mortality)

  def mortality_event(self, t):
    self.alive = False
    self.time_mortality = self.age + t 
    self.finish()

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard, n):
    # initialise population
    self.population = [ Person(mortality_hazard) for _ in range(n) ]
    self.life_expectancy = 0.0

  def age(self):
    [p.inc_age() for p in self.population]

  def calc_life_expectancy(self):  
    # compute mean
    le = 0.0
    n = 0
    for p in self.population:
      if not p.alive:
        le = le + p.time_mortality
        n = n + 1
    return le / n
    #self.life_expectancy = sum([p.time_mortality for p in self.population]) / len(self.population)
    #return self.life_expectancy

  def prop_alive(self):  
    # # compute mean
    neworder.log(sum([p.alive for p in self.population]) / len(self.population))
    return True

  # def fraction_alive(self, t):
  #   count = sum([p.state(t) for p in self.population]) / len(self.population)
