
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import neworder
import ethpop
import animation

# A more "pythonic" approach using pandas DataFrames

class People():
  """ A simple aggregration of Persons each represented as a row in a data frame """
  def __init__(self, fertility_hazard_file, mortality_hazard_file, lad, ethnicity, n):
    # initialise cohort      
    # assert False
    # filter by location, ethnicity and gender
    self.fertility_hazard = ethpop.create(pd.read_csv(fertility_hazard_file), lad, truncate85=False).reset_index()

    self.fertility_hazard = self.fertility_hazard[(self.fertility_hazard.NewEthpop_ETH==ethnicity) 
                                                & (self.fertility_hazard.DC1117EW_C_SEX==2)]

    self.mortality_hazard = ethpop.create(pd.read_csv(mortality_hazard_file), lad, truncate85=False).reset_index()

    self.mortality_hazard = self.mortality_hazard[(self.mortality_hazard.NewEthpop_ETH==ethnicity) 
                                                & (self.mortality_hazard.DC1117EW_C_SEX==2)]

    # store the largest age we have a rate for 
    self.max_rate_age = int(max(self.mortality_hazard.DC1117EW_C_AGE) - 1)

    #neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(data={"Parity": np.zeros(n, dtype=int),
                                         "TimeOfBaby1": neworder.far_future(),
                                         "TimeOfDeath": np.zeros(n, dtype=float)})

  def plot(self, filename=None):
    # dump the population out
    #self.population.to_csv(filename, index=False)

    buckets = range(self.max_rate_age + 10)

    # add some time on the end to capture (most of) those who die over the max simulation age
    plt.hist(self.population.TimeOfDeath, buckets, color='black')
    b = [ self.population.TimeOfBaby1[~np.isnan(self.population.TimeOfBaby1)], 
          self.population.TimeOfBaby2[~np.isnan(self.population.TimeOfBaby2)],
          self.population.TimeOfBaby3[~np.isnan(self.population.TimeOfBaby3)],
          self.population.TimeOfBaby4[~np.isnan(self.population.TimeOfBaby4)] ]
    plt.hist(b, buckets, stacked=True)
    #plt.savefig("./doc/examples/img/competing_hist_100k.png")
    plt.show()

  def age(self):
    # sample times

    self.population["TimeOfDeath"] = neworder.mc.first_arrival(self.mortality_hazard.Rate.values, 1.0, len(self.population))

    #neworder.timeline.dt()
    births = neworder.mc.arrivals(self.fertility_hazard.Rate.values, 1.0, 0.75, len(self.population))
    
    #neworder.log(births)
    for i in range(births.shape[1]):
      col = "TimeOfBaby" + str(i+1)
      self.population[col] = births[:,i]
      # remove births that would have occured after death
      self.population.loc[self.population[col] > self.population.TimeOfDeath, col] = neworder.never() 
      self.population.Parity = self.population.Parity + ~np.isnan(self.population[col])


  def stats(self):  
    # # compute mean
    neworder.log("birth rate = %f" % np.mean(self.population.Parity))
    neworder.log("pct mother = %f" % (100.0 * np.mean(self.population.Parity > 0)))
    neworder.log("life exp. = %f" % np.mean(self.population.TimeOfDeath))
    #return True
