
import numpy as np
import pandas as pd
import neworder
import ethpop

class People(neworder.Model):
  """ A simple aggregration of Persons each represented as a row in a data frame """
  def __init__(self, mortality_hazard_file, n, max_age):
    # This is case-based model the timeline refers to the age of the cohort
    timeline = neworder.Timeline(0.0, max_age, [int(max_age)])
    super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_stream)
    # initialise cohort
    # filter by location, ethnicity and gender
    self.mortality_hazard = ethpop.create(pd.read_csv(mortality_hazard_file), "E09000030", truncate85=False).reset_index()

    self.mortality_hazard = self.mortality_hazard[(self.mortality_hazard.NewEthpop_ETH=="WBI")
                                                & (self.mortality_hazard.DC1117EW_C_SEX==1)]

    # store the largest age we have a rate for
    self.max_rate_age = max(self.mortality_hazard.DC1117EW_C_AGE) - 1

    #neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(data={"Alive": np.full(n, True),
                                         "Age": np.zeros(n),
                                         "TimeOfDeath": np.zeros(n)})

    self.max_age = max_age

  def step(self):
    # kill off some people
    self.die()
    # age the living only
    alive = self.population.loc[self.population.Alive].index
    self.population.loc[alive, "Age"] = self.population.loc[alive, "Age"] + self.timeline().dt()

  def check(self):
    self.prop_alive()
    return True

  def checkpoint(self):
    neworder.log(self.calc_life_expectancy())

  def die(self):
    # using indexes to subset data as cannot store a reference to a subset of the dataframe (it just copies)

    # first filter out the already dead
    alive = self.population.loc[self.population.Alive].index
    # sample time of death
    r = self.mc().stopping(self.mortality_hazard.Rate.values[min(self.timeline().index()-1, self.max_rate_age)], len(alive))
    # select if death happens before next timestep...
    dt = self.timeline().dt()
    # at final timestep everybody dies (at some later time) so dt is infinite
    if self.timeline().time() == self.max_age:
      dt = neworder.time.far_future()
    #
    newly_dead = alive[r<dt]

    # kill off those who die before next timestep
    self.population.loc[newly_dead, "Alive"] = False
    self.population.loc[newly_dead, "TimeOfDeath"] = self.population.loc[newly_dead, "Age"] + r[r<dt]

  def calc_life_expectancy(self):
    # ensure all people have died
    assert np.sum(self.population.Alive) == 0

    # in this case we can also compute the mortality directly by modelling a non-homogeneous Poisson process
    # using the Lewis-Shedler algorithm
    self.population["TimeOfDeathNHPP"] = self.mc().first_arrival(self.mortality_hazard.Rate.values, self.timeline().dt(), len(self.population))

    # compare the discrete simulation value against the more direct computation
    neworder.log("%f vs %f" % (np.mean(self.population.TimeOfDeath), np.mean(self.population.TimeOfDeathNHPP)))
    return np.mean(self.population.TimeOfDeath)

  def prop_alive(self):
    # # compute mean
    neworder.log("pct alive = %f" % (100.0 * np.mean(self.population.Alive)))
