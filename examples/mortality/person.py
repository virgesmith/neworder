import pandas as pd
from matplotlib import pyplot as plt

import neworder
import ethpop

class Person():

  def __init__(self):
    """ Person::Start() """
    self.alive = True
    self.age = 0.0
    #self.time = 0.0

    self.time_mortality = neworder.TIME_INFINITY

  def finish(self):
    """ Person::Finish() """
    pass

  # def state(self, t):
  #   """ Returns the person's state (alive/dead) at age t """
  #   return self.alive

  def inc_age(self, mortality_hazard):
    self.time_mortality_event(mortality_hazard)
    if self.alive:
      self.age = self.age + neworder.timestep

  def time_mortality_event(self, mortality_hazard):
    """ TIME Person::timeMortalityEvent() """
    t = neworder.stopping(mortality_hazard.Rate.values[min(neworder.timeindex-1, 101)], 1)[0]
    if t < neworder.timestep or self.age >= neworder.timespan[-1] - neworder.timestep:
      self.mortality_event(t)
    #neworder.log("TOD=%f" % self.time_mortality)

  def mortality_event(self, t):
    self.alive = False
    self.time_mortality = self.age + t 
    #neworder.log("died @ %f aged %f" % (t, self.age))
    self.finish()

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard_file, n):

    self.mortality_hazard = ethpop.create(pd.read_csv(mortality_hazard_file), "E09000030", truncate85=False).reset_index()

    self.mortality_hazard = self.mortality_hazard[(self.mortality_hazard.NewEthpop_ETH=="WBI") 
                                                & (self.mortality_hazard.DC1117EW_C_SEX==1)]

    self.max_rate_age = max(self.mortality_hazard.DC1117EW_C_AGE) - 1
    # initialise population
    self.population = [ Person() for _ in range(n) ]
    self.life_expectancy = 0.0

  def age(self):
    [p.inc_age(self.mortality_hazard) for p in self.population]

  def calc_life_expectancy(self):  

    #neworder.log("prop_alive=%f" % self.prop_alive())
    assert sum([p.alive for p in self.population]) == 0
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

  def plot(self, filename=None):
    # dump the population out
    #self.population.to_csv(filename, index=False)
    plt.hist([p.time_mortality for p in self.population], self.max_rate_age)
    plt.show()
