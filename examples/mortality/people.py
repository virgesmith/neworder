
import numpy as np
import pandas as pd
import neworder


# !disc_ctor!
class PeopleDiscrete(neworder.Model):
  """ Persons sampled each represented as a row in a data frame """
  def __init__(self, mortality_hazard_file, n, max_age):
    # This is case-based model the timeline refers to the age of the cohort
    timeline = neworder.LinearTimeline(0.0, max_age, int(max_age))
    super().__init__(timeline, neworder.MonteCarlo.deterministic_identical_stream)

    # initialise cohort
    self.mortality_hazard = pd.read_csv(mortality_hazard_file)

    # store the largest age we have a rate for
    self.max_rate_age = max(self.mortality_hazard.DC1117EW_C_AGE) - 1

    # neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(index=neworder.df.unique_index(n),
                                   data={"alive": True,
                                         "age": 0.0,
                                         "age_at_death": neworder.time.far_future()})

    self.max_age = max_age
# !disc_ctor!

  # !disc_step!
  def step(self):
    # kill off some people
    self.die()
    # age the living only
    alive = self.population.loc[self.population.alive].index
    self.population.loc[alive, "age"] = self.population.loc[alive, "age"] + self.timeline.dt()
  # !disc_step!

  def check(self):
    neworder.log("pct alive = %f" % (100.0 * np.mean(self.population.alive)))
    return True

  # !disc_finalise!
  def finalise(self):
    # kill off any survivors
    self.die()
    assert np.sum(self.population.alive) == 0
    # the calc life expectancy
    self.life_expectancy = np.mean(self.population.age_at_death)
  # !disc_finalise!

  def die(self):
    # using indexes to subset data as cannot store a reference to a subset of the dataframe (it just copies)

    # first filter out the already dead
    alive = self.population.loc[self.population.alive].index
    # sample time of death
    r = self.mc.stopping(self.mortality_hazard.Rate.values[min(self.timeline.index(), self.max_rate_age)], len(alive))
    # select if death happens before next timestep...
    dt = self.timeline.dt()
    # at final timestep everybody dies (at some later time) so dt is infinite
    if self.timeline.time() == self.max_age:
      dt = neworder.time.far_future()
    newly_dead = alive[r < dt]

    # kill off those who die before next timestep
    self.population.loc[newly_dead, "alive"] = False
    # and set the age at death according to the stopping time above
    self.population.loc[newly_dead, "age_at_death"] = self.population.loc[newly_dead, "age"] + r[r < dt]


# !cont_ctor!
class PeopleContinuous(neworder.Model):
  """ Persons sampled each represented as a row in a data frame """
  def __init__(self, mortality_hazard_file, n, dt):
    # Direct sampling doesnt require a timeline
    super().__init__(neworder.NoTimeline(), neworder.MonteCarlo.deterministic_identical_stream)
    # initialise cohort
    self.mortality_hazard = pd.read_csv(mortality_hazard_file)

    # store the largest age we have a rate for
    self.max_rate_age = max(self.mortality_hazard.DC1117EW_C_AGE) - 1

    # neworder.log(self.mortality_hazard.head())
    self.population = pd.DataFrame(index=neworder.df.unique_index(n),
                                   data={"age_at_death": neworder.time.far_future()})

    # the time interval of the mortality data values
    self.dt = dt
# !cont_ctor!

  # !cont_step!
  def step(self):
    self.population.age_at_death = self.mc.first_arrival(self.mortality_hazard.Rate.values, self.dt, len(self.population))
  # !cont_step!

  # !cont_check!
  def check(self):
    # ensure all times of death are finite
    return self.population.age_at_death.isnull().sum() == 0
  # !cont_check!

  # !cont_finalise!
  def finalise(self):
    self.life_expectancy = np.mean(self.population.age_at_death)
  # !cont_finalise!
