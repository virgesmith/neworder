
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import neworder
import ethpop
import animation

# A more "pythonic" approach using pandas DataFrames

class People():
  """ A simple aggregration of Persons each represented as a row in a data frame """
  def __init__(self, fertility_hazard_file, mortality_hazard_file, n):
    # initialise cohort      
    # assert False
    # filter by location, ethnicity and gender
    self.fertility_hazard = ethpop.create(pd.read_csv(fertility_hazard_file), "E09000030", truncate85=False).reset_index()

    self.fertility_hazard = self.fertility_hazard[(self.fertility_hazard.NewEthpop_ETH=="WBI") 
                                                & (self.fertility_hazard.DC1117EW_C_SEX==2)]

    self.mortality_hazard = ethpop.create(pd.read_csv(mortality_hazard_file), "E09000030", truncate85=False).reset_index()

    self.mortality_hazard = self.mortality_hazard[(self.mortality_hazard.NewEthpop_ETH=="WBI") 
                                                & (self.mortality_hazard.DC1117EW_C_SEX==2)]

    # store the largest age we have a rate for 
    self.max_rate_age = int(max(self.mortality_hazard.DC1117EW_C_AGE) - 1)

    #neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(data={"Parity": np.zeros(n, dtype=int),
                                         "TimeOfBaby1": neworder.far_future(),
                                         "TimeOfDeath": np.zeros(n, dtype=float)})

  def plot(self, filename=None):
    # dump the population out
    self.population.to_csv(filename, index=False)

    #plt.plot(self.mortality_hazard.DC1117EW_C_AGE, self.mortality_hazard.Rate)                                            

    # y1, x1 = np.histogram(self.population.TimeOfDeathNHPP, int(max(self.population.Age)))
    # plt.plot(x1[1:], y1)
    #y2, x2 = np.histogram(self.population.TimeOfDeath, self.max_rate_age)
    #plt.plot(x2[1:], y2)
    #animation.Hist(self.population.TimeOfDeath, self.max_rate_age, filename)
    plt.hist(self.population.TimeOfDeath, self.max_rate_age)
    plt.hist(self.population.TimeOfBaby1, self.max_rate_age)
    plt.hist(self.population.TimeOfBaby2, self.max_rate_age)
    plt.hist(self.population.TimeOfBaby3, self.max_rate_age)
    plt.hist(self.population.TimeOfBaby4, self.max_rate_age)
    plt.show()

  def age(self):
    # sample times

    self.population["TimeOfDeath"] = neworder.stopping_nhpp(self.mortality_hazard.Rate.values, neworder.timestep, len(self.population))
    # hack to account for possibility of childlessness
    self.fertility_hazard.Rate.values[-1] = 1.0
    #self.population["TimeOfBaby1"] = neworder.stopping_nhpp(self.fertility_hazard.Rate.values, neworder.timestep, len(self.population))
    #neworder.log(neworder.never())
    #self.population.loc[self.population["TimeOfBaby1"] >= 100.0, "TimeOfBaby1"] = neworder.never()

    births = neworder.stopping_nhpp_multi(self.fertility_hazard.Rate.values, neworder.timestep, 0.75, len(self.population))
    
    #neworder.log(births[:,0])
    for i in range(births.shape[1]): 
      self.population["TimeOfBaby" + str(i+1)] = births[:,i]

    
    #self.population.Parity = int(self.population.TimeOfBaby1 != neworder.never())# & (self.population.TimeOfBaby1 < self.population.TimeOfDeath))

  def calc_life_expectancy(self):  

    #neworder.log("%f vs %f" % (np.mean(self.population.TimeOfDeath), np.mean(self.population.TimeOfDeath)))
    return np.mean(self.population.TimeOfDeath)

  def prop_mother(self):  
    # # compute mean
    #neworder.log("pct mother = %f" % (100.0 * np.mean(self.population.Parity > 0)))
    return True
