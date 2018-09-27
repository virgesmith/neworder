
import numpy as np
import pandas as pd 
import neworder
## for TIME_INFINITY (could be passed as an)
#import config

# class Person():

#   def __init__(self, mortality_hazard):
#     """ Person::Start() """

#     self.alive = True
#     self.age = 0.0
#     #self.time = 0.0
#     self.mortality_hazard = mortality_hazard
#     self.time_mortality = config.TIME_INFINITY

#   def __del__(self):
#     """ Person::Finish() """

#   def state(self, t):
#     """ Returns the person's state (alive/dead) at age t """
#     return self.alive

#   def inc_age(self):
#     self.time_mortality_event()
#     if self.alive:
#       self.age = self.age + neworder.timestep

#   def time_mortality_event(self):
#     """ TIME Person::timeMortalityEvent() """
#     t = neworder.stopping(config.mortality_hazard, 1)[0]
#     if t < neworder.timestep or self.age >= neworder.timespan[-2]:
#       self.mortality_event(t)
#     #neworder.log("TOD=%f" % self.time_mortality)

#   def mortality_event(self, t):
#     self.alive = False
#     self.time_mortality = self.age + t 
#     # Person.__del__(self)

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard, n):
    # initialise population
      
    self.population = pd.DataFrame(data={"MortalityRate": np.full(n, mortality_hazard), 
                                         "Alive": np.full(n, True),
                                         "Age": np.zeros(n), 
                                         "TimeOfDeath": np.zeros(n)})
    #self.population = [ Person(mortality_hazard) for _ in range(n) ]
    #self.life_expectancy = 0.0
    neworder.log(self.population.head())

  def age(self):
    # take the living

    alive = self.population.loc[self.population.Alive]
    alive.loc[neworder.stopping_v(alive.MortalityRate.values) < neworder.timestep, "Alive"] = False

    #self.population.loc[neworder.stopping_v(self.population.MortalityRate.values) < neworder.timestep, "Alive"] = False
    #neworder.log(self.population.loc[self.population.Alive == False])
    #.Alive = False
    # #alive = self.population.loc[self.population.Alive, :]
    # neworder.log(alive.head())
    # alive.Alive = False
    # self.population.loc[self.population.Alive, :].Alive = False
    # # # sample time of death
    # alive.TimeOfDeath = alive.Age + neworder.stopping_v(alive.MortalityRate.values)
    # alive.loc[alive.TimeOfDeath < alive.Age + neworder.timestep].Alive = False

    # #alive.loc[u < neworder.timestep].Alive = False
    # #alive.loc[u < neworder.timestep].TimeOfDeath = alive[u < neworder.timestep].Age + u
    # # mark dead if before next timestep
    # #[p.inc_age() for p in self.population]
    # alive = self.population[self.population.Alive]
    # #alive = self.population.loc[self.population.Alive]
    # alive.Age = alive.Age + 1
    neworder.log(self.population.head(10))

  def calc_life_expectancy(self):  
    pass
    # compute mean
    # le = 0.0
    # n = 0
    # for p in self.population:
    #   if not p.alive:
    #     le = le + p.time_mortality
    #     n = n + 1
    # return le / n
    # #self.life_expectancy = sum([p.time_mortality for p in self.population]) / len(self.population)
    # #return self.life_expectancy

  def prop_alive(self):  
    # # compute mean
    neworder.log(np.mean(self.population.Alive))
    return False
    #pass

  # def fraction_alive(self, t):
  #   count = sum([p.state(t) for p in self.population]) / len(self.population)
