# the framework must be explicitly imported
import neworder
import numpy as np

class Person():
  """ 
  MODGEN equivalent: actor Person {...}
  Represents a single individual 
  """
  def __init__(self, mortality_hazard):
    """ MODGEN equivalent: Person::Start() """
    self.alive = True
    # MODGEN would automatically create time and age, though they are not needed for this simple example
    self.mortality_hazard = mortality_hazard
    self.time_mortality = -1.0 # to be computed later

  def finish(self):
    """ MODGEN equivalent: Person::Finish() """
    # nothing required here

  def state(self, t):
    """ Returns the person's state (alive/dead) at age t """
    return True if self.time_mortality > t else False

  def time_mortality_event(self, mc):
    """ MODGEN equivalent: TIME Person::timeMortalityEvent() """
    self.time_mortality = mc.stopping(self.mortality_hazard, 1)[0]

  def mortality_event(self):
    """ MODGEN equivalent: void Person::MortalityEvent() """
    # NB this is not used in this implementation
    self.alive = False

class People(neworder.Model):
  """ A model containing an aggregration of Person objects """
  def __init__(self, mortality_hazard, n):
    
    # initialise base model with a nondeterministic seed results will vary (slightly)
    super().__init__(neworder.Timeline.null(), neworder.MonteCarlo.nondeterministic_stream)

    # initialise population
    self.population = [ Person(mortality_hazard) for _ in range(n) ]
    neworder.log("created %d individuals" % n)

  def step(self):
    # sample each person's age at death. since everyone 
    # (this is a hopelessly inefficient implementation when everyone has the same hazard rate)
    [p.time_mortality_event(self.mc()) for p in self.population]

  def checkpoint(self):
    # compute mean sampled life expectancy against theoretical
    sample_le = sum([p.time_mortality for p in self.population]) / len(self.population)
    actual_le = 1.0 / self.population[0].mortality_hazard
    error = sample_le - actual_le
    neworder.log("Life expectancy = %.2f years (sampling error=%.2f years)" % (sample_le, error))

  def alive(self, t):
    return np.mean([p.state(t) for p in self.population])

