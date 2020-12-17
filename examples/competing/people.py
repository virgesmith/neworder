
import numpy as np
import pandas as pd
import neworder as no
#import ethpop

class People(no.Model):
  """ A simple aggregration of Persons each represented as a row in a data frame """
  def __init__(self, dt, fertility_hazard_file, mortality_hazard_file, n):

    super().__init__(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

    self.dt = dt # time resolution of fertility/mortality data

    self.fertility_hazard = pd.read_csv(fertility_hazard_file)
    self.mortality_hazard = pd.read_csv(mortality_hazard_file)

    # store the largest age we have a rate for
    self.max_rate_age = int(max(self.mortality_hazard.DC1117EW_C_AGE) - 1)

    # initialise cohort
    self.population = pd.DataFrame(index=no.df.unique_index(n),
                                   data={"parity": 0,
                                         "time_of_death": no.time.far_future()})

  def step(self):
    # sample deaths
    self.population["time_of_death"] = self.mc().first_arrival(self.mortality_hazard.Rate.values, self.dt, len(self.population))

    # sample (multiple) births with events at least 9 months apart
    births = self.mc().arrivals(self.fertility_hazard.Rate.values, self.dt, len(self.population), 0.75)

    # the number of columns is governed by the maximum number of arrivals in the births data
    for i in range(births.shape[1]):
      col = "time_of_baby_" + str(i+1)
      self.population[col] = births[:,i]
      # remove births that would have occured after death
      self.population.loc[self.population[col] > self.population.time_of_death, col] = no.time.never()
      self.population.parity = self.population.parity + ~no.time.isnever(self.population[col].values)

  def finalise(self):
    # compute means
    no.log("birth rate = %f" % np.mean(self.population.parity))
    no.log("percentage mothers = %f" % (100.0 * np.mean(self.population.parity > 0)))
    no.log("life expexctancy = %f" % np.mean(self.population.time_of_death))
