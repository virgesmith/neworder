
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import neworder
import ethpop
import animation

# A more "pythonic" approach using pandas DataFrames

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard_file, n):
    # initialise cohort      
    # assert False
    # filter by location, ethnicity and gender
    self.mortality_hazard = ethpop.create(pd.read_csv(mortality_hazard_file), "E09000030").reset_index()

    neworder.log(self.mortality_hazard.head())
    self.mortality_hazard = self.mortality_hazard[(self.mortality_hazard.NewEthpop_ETH=="WBI") 
                                                & (self.mortality_hazard.DC1117EW_C_SEX==1)]
    neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(data={"Alive": np.full(n, True),
                                         "Age": np.zeros(n), 
                                         "TimeOfDeath": np.zeros(n)})
    #neworder.log(self.population.head())
    #plt.ion() # interactive on stops blocking


  def plot(self, hold=False):
    # dump the population out
    #self.population.to_csv(filename, index=False)
    # if hold:
    #   plt.ioff()
    # else:
    #   plt.cla()
    y,_,_ = plt.hist(self.population[self.population.Alive == False].TimeOfDeath, 100)
    # neworder.log(y)
    a = animation.Animation(np.array(range(len(y))), y)
    # plt.show()
    # plt.pause(0.1)

  def die(self):
    # using indexes to subset data as cannot store a reference to a subset of the dataframe (it just copies)

    # first filter out the already dead
    alive = self.population.loc[self.population.Alive].index
    # sample time of death
    r = neworder.stopping(self.mortality_hazard.Rate.values[min(neworder.timeindex-1, 85)], len(alive))
    # select if death happens before next timestep...
    dt = neworder.timestep
    # at final timestep everybody dies (at some later time) so dt is infinite
    if neworder.time == neworder.MAX_AGE:
      dt = neworder.TIME_INFINITY
    # 
    newly_dead = alive[r<dt]

    # kill off those who die before next timestep
    self.population.ix[newly_dead, "Alive"] = False
    self.population.ix[newly_dead, "TimeOfDeath"] = self.population.ix[newly_dead, "Age"] + np.clip(r[r<dt], 0.0, 15.0)

  def age(self):
    # kill off some people
    self.die()

    # age the living only
    alive = self.population.loc[self.population.Alive].index
    self.population.ix[alive, "Age"] = self.population.ix[alive, "Age"] + neworder.timestep

  def calc_life_expectancy(self):  
    # ensure all people have died 
    #self.dump("./population.csv")

    assert np.sum(self.population.Alive) == 0
    return np.mean(self.population.TimeOfDeath)

  def prop_alive(self):  
    # # compute mean
    neworder.log("pct alive = %f" % (100.0 * np.mean(self.population.Alive)))
    return True
