
import numpy as np
import pandas as pd
import neworder

class PeopleDiscrete(neworder.Model):
  """ Persons sampled each represented as a row in a data frame """
  def __init__(self, mortality_hazard_file, n, max_age):
    # This is case-based model the timeline refers to the age of the cohort
    timeline = neworder.Timeline(0.0, max_age, [int(max_age)])
    super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_stream)

    # initialise cohort
    self.mortality_hazard = pd.read_csv(mortality_hazard_file)

    # store the largest age we have a rate for
    self.max_rate_age = max(self.mortality_hazard.DC1117EW_C_AGE) - 1

    #neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(data={"alive": np.full(n, True),
                                         "age": np.zeros(n),
                                         "age_at_death": np.zeros(n)})

    self.max_age = max_age

  def step(self):
    # kill off some people
    self.die()
    # age the living only
    alive = self.population.loc[self.population.alive].index
    self.population.loc[alive, "age"] = self.population.loc[alive, "age"] + self.timeline().dt()

  def check(self):
    neworder.log("pct alive = %f" % (100.0 * np.mean(self.population.alive)))
    return True

  def checkpoint(self):
    pass

  def die(self):
    # using indexes to subset data as cannot store a reference to a subset of the dataframe (it just copies)

    # first filter out the already dead
    alive = self.population.loc[self.population.alive].index
    # sample time of death
    r = self.mc().stopping(self.mortality_hazard.Rate.values[min(self.timeline().index()-1, self.max_rate_age)], len(alive))
    # select if death happens before next timestep...
    dt = self.timeline().dt()
    # at final timestep everybody dies (at some later time) so dt is infinite
    if self.timeline().time() == self.max_age:
      dt = neworder.time.far_future()
    newly_dead = alive[r<dt]

    # kill off those who die before next timestep
    self.population.loc[newly_dead, "alive"] = False
    self.population.loc[newly_dead, "age_at_death"] = self.population.loc[newly_dead, "age"] + r[r<dt]

  def calc_life_expectancy(self):
    # ensure all people have died
    assert np.sum(self.population.alive) == 0
    return np.mean(self.population.age_at_death)


class PeopleContinuous(neworder.Model):
  """ Persons sampled each represented as a row in a data frame """
  def __init__(self, mortality_hazard_file, n, max_age):
    # Direct sampling doesnt require a timeline
    super().__init__(neworder.Timeline.null(), neworder.MonteCarlo.deterministic_identical_stream)
    # initialise cohort
    self.mortality_hazard = pd.read_csv(mortality_hazard_file)

    # store the largest age we have a rate for
    self.max_rate_age = max(self.mortality_hazard.DC1117EW_C_AGE) - 1

    #neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(data={"age_at_death": np.zeros(n)})

    self.max_age = max_age

  def step(self):
    self.population.age_at_death = self.mc().first_arrival(self.mortality_hazard.Rate.values, 1.0, len(self.population))

  def check(self):
    # ensure all times of death are finite 
    return self.population.age_at_death.isnull().sum() == 0    

  def checkpoint(self):
    pass

  def calc_life_expectancy(self):
    return np.mean(self.population.age_at_death)
