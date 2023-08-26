# the framework must be explicitly imported
import neworder
import numpy as np

# !person!
class Person():
  """
  MODGEN equivalent: actor Person {...}
  Represents a single individual
  """
  def __init__(self, mortality_hazard: float) -> None:
    """ MODGEN equivalent: Person::Start() """
    self.alive = True
    # MODGEN would automatically create time and age, though they are not needed for this simple example
    self.mortality_hazard = mortality_hazard
    self.time_mortality = neworder.time.NEVER # to be computed later

  def finish(self) -> None:
    """ MODGEN equivalent: Person::Finish() """
    # nothing required here

  def state(self, t: float) -> bool:
    """ Returns the person's state (alive/dead) at age t """
    return self.time_mortality > t

  def time_mortality_event(self, mc: neworder.MonteCarlo) -> float:
    """ MODGEN equivalent: TIME Person::timeMortalityEvent() """
    if neworder.time.isnever(self.time_mortality):
      self.time_mortality = mc.stopping(self.mortality_hazard, 1)[0]
    return self.time_mortality

  def mortality_event(self) -> None:
    """ MODGEN equivalent: void Person::MortalityEvent() """
    # NB this is not used in this implementation
    self.alive = False
# !person!

# !constructor!
class People(neworder.Model):
  """ A model containing an aggregration of Person objects """
  def __init__(self, mortality_hazard: float, n: int) -> None:

    # initialise base model with a nondeterministic seed results will vary (slightly)
    super().__init__(neworder.NoTimeline(), neworder.MonteCarlo.nondeterministic_stream)

    # initialise population
    self.population = [Person(mortality_hazard) for _ in range(n)]
    neworder.log("created %d individuals" % n)
# !constructor!

# !step!
  def step(self) -> None:
    # sample each person's age at death.
    # (this is not an efficient implementation when everyone has the same hazard rate)
    [p.time_mortality_event(self.mc) for p in self.population]
# !step!

# !finalise!
  def finalise(self) -> None:
    # compute mean sampled life expectancy against theoretical
    sample_le = sum([p.time_mortality for p in self.population]) / len(self.population)
    actual_le = 1.0 / self.population[0].mortality_hazard
    error = sample_le - actual_le
    neworder.log("Life expectancy = %.2f years (sampling error=%.2f years)" % (sample_le, error))
# !finalise!

# !alive!
  def alive(self, t: float) -> float:
    return np.mean([p.state(t) for p in self.population])
# !alive!

