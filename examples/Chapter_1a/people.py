
import numpy as np
import pandas as pd 
import neworder

# A more "pythonic" approach using pandas DataFrames

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard, n):
    # initialise cohort      
    self.population = pd.DataFrame(data={"MortalityRate": np.full(n, mortality_hazard), 
                                         "Alive": np.full(n, True),
                                         "Age": np.zeros(n), 
                                         "TimeOfDeath": np.zeros(n)})
    neworder.log(self.population.head())

  def die(self):
    # using indexes to subset data as cannot store a reference to a subset of the dataframe (it just copies)

    # first filter out the already dead
    alive = self.population.loc[self.population.Alive].index
    # sample time of death
    r = neworder.stopping_v(self.population.ix[alive, "MortalityRate"].values)
    # select if death happens before next timestep...
    dt = neworder.timestep
    # at final timestep everybody dies (at some later time) so dt is infinite
    if neworder.time == neworder.MAX_AGE:
      dt = neworder.TIME_INFINITY
    # 
    newly_dead = alive[r<dt]

    # kill off those who die before next timestep
    self.population.ix[newly_dead, "Alive"] = False
    self.population.ix[newly_dead, "TimeOfDeath"] = self.population.ix[newly_dead, "Age"] + r[r<dt]

  def age(self):
    # kill off some people
    self.die()

    # age the living only
    alive = self.population.loc[self.population.Alive].index
    self.population.ix[alive, "Age"] = self.population.ix[alive, "Age"] + neworder.timestep

  def calc_life_expectancy(self):  
    # ensure all people have died 
    assert np.sum(self.population.Alive) == 0
    return np.mean(self.population.TimeOfDeath)

  def prop_alive(self):  
    # # compute mean
    neworder.log("pct alive = %f" % (100.0 * np.mean(self.population.Alive)))
    return True
