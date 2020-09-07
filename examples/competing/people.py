
import numpy as np
import pandas as pd
import neworder as no
import ethpop

class People(no.Model):
  """ A simple aggregration of Persons each represented as a row in a data frame """
  def __init__(self, dt, fertility_hazard_file, mortality_hazard_file, lad, ethnicity, n):

    super().__init__(no.Timeline.null(), no.MonteCarlo.deterministic_identical_stream)

    self.dt = dt # time resolution of fertility/mortality data

    # initialise cohort
    # filter by location, ethnicity and gender
    self.fertility_hazard = ethpop.create(pd.read_csv(fertility_hazard_file), lad, truncate85=False).reset_index()

    self.fertility_hazard = self.fertility_hazard[(self.fertility_hazard.NewEthpop_ETH==ethnicity)
                                                & (self.fertility_hazard.DC1117EW_C_SEX==2)]

    self.mortality_hazard = ethpop.create(pd.read_csv(mortality_hazard_file), lad, truncate85=False).reset_index()

    self.mortality_hazard = self.mortality_hazard[(self.mortality_hazard.NewEthpop_ETH==ethnicity)
                                                & (self.mortality_hazard.DC1117EW_C_SEX==2)]

    # store the largest age we have a rate for
    self.max_rate_age = int(max(self.mortality_hazard.DC1117EW_C_AGE) - 1)

    self.population = pd.DataFrame(data={"Parity": np.zeros(n, dtype=int),
                                         "TimeOfDeath": no.time.far_future()})

  def step(self):
    self.population["TimeOfDeath"] = self.mc().first_arrival(self.mortality_hazard.Rate.values, self.dt, len(self.population))

    births = self.mc().arrivals(self.fertility_hazard.Rate.values, self.dt, 0.75, len(self.population))

    # the number of columns is governed by the maximum number of arrivals in the births data 
    for i in range(births.shape[1]):
      col = "TimeOfBaby" + str(i+1)
      self.population[col] = births[:,i]
      # remove births that would have occured after death
      self.population.loc[self.population[col] > self.population.TimeOfDeath, col] = no.time.never()
      self.population.Parity = self.population.Parity + ~no.time.isnever(self.population[col].values)

  def checkpoint(self):
    # nothing more to do
    pass
