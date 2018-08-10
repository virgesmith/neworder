
import neworder

class Person():

  def __init__(self):
    """ Person::Start() """
    self.alive = True
    self.age = 0.0
    self.time = 0.0
    self.time_mortality = neworder.time_infinity

  def __del__(self):
    """ Person::Finish() """

  def state(self, t):
    """ Returns the person's state (alive/dead) at time t """
    return True if self.time_mortality > t else False

  def time_mortality_event(self):
    """ TIME Person::timeMortalityEvent() """
    self.time_mortality = neworder.stopping(neworder.mortality_hazard, 1)[0]
    #neworder.log("TOD=%d" % self.time_mortality)

  def mortality_event(self):
    self.alive = False
    # Person.__del__(self)

class People():
  """ A simple aggregration of Person """
  def __init__(self, n):
    # initialise population
    self.population = [ Person() for _ in range(n) ]

  def mean_lifespan(self):
    expectancy = sum([p.time_mortality for p in self.population]) / len(self.population)
    neworder.log("LE=%d" % expectancy)
    return expectancy > 69.0 and expectancy < 73.0
