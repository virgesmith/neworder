# the framework must be explicitly imported
import neworder

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

  def time_mortality_event(self):
    """ MODGEN equivalent: TIME Person::timeMortalityEvent() """
    self.time_mortality = neworder.stopping(self.mortality_hazard, 1)[0]

  def mortality_event(self):
    """ MODGEN equivalent: void Person::MortalityEvent() """
    self.alive = False

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard, n):
    # initialise population
    self.population = [ Person(mortality_hazard) for _ in range(n) ]
    neworder.log("created %d individuals" % n)

  def sample_mortality(self):
    # sample age at death for each member of the population
    [p.time_mortality_event() for p in self.population]

  def calc_life_expectancy(self):
    # compute mean sampled life expectancy against theoretical
    sample_le = sum([p.time_mortality for p in self.population]) / len(self.population)
    actual_le = 1.0 / self.population[0].mortality_hazard
    error = sample_le - actual_le
    neworder.log("Life expectancy = %.2f years (sampling error=%f)" % (sample_le, error))

