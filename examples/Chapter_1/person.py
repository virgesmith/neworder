
import neworder
# for TIME_INFINITY (could be passed as an)
import config
#from config import TIME_INFINITY

class Person():

  def __init__(self, mortality_hazard):
    """ Person::Start() """
    self.alive = True
    self.age = 0.0
    self.time = 0.0
    self.mortality_hazard = mortality_hazard
    self.time_mortality = config.TIME_INFINITY

  def __del__(self):
    """ Person::Finish() """

  def state(self, t):
    """ Returns the person's state (alive/dead) at age t """
    return True if self.time_mortality > t else False

  def time_mortality_event(self):
    """ TIME Person::timeMortalityEvent() """
    self.time_mortality = neworder.stopping(config.mortality_hazard, 1)[0]
    #neworder.log("TOD=%f" % self.time_mortality)

  def mortality_event(self):
    self.alive = False
    # Person.__del__(self)

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard, n):
    # initialise population
    self.population = [ Person(mortality_hazard) for _ in range(n) ]
    #self.life_expectancy = 0.0

  def calc_life_expectancy(self):
    # sample age at death for population
    [p.time_mortality_event() for p in self.population]
    # compute mean
    self.life_expectancy = sum([p.time_mortality for p in self.population]) / len(self.population)

  # def fraction_alive(self, t):
  #   count = sum([p.state(t) for p in self.population]) / len(self.population)
