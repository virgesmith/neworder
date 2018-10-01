
import neworder

class Person():

  def __init__(self, mortality_hazard):
    """ Person::Start() """
    self.alive = True
    self.age = 0.0
    #self.time = 0.0
    self.mortality_hazard = mortality_hazard
    self.time_mortality = neworder.TIME_INFINITY

  def __del__(self):
    """ Person::Finish() """

  def state(self, t):
    """ Returns the person's state (alive/dead) at age t """
    return self.alive

  def inc_age(self):
    self.time_mortality_event()
    if self.alive:
      self.age = self.age + neworder.timestep

  def time_mortality_event(self):
    """ TIME Person::timeMortalityEvent() """
    t = neworder.stopping(self.mortality_hazard, 1)[0]
    if t < neworder.timestep or self.age >= neworder.timespan[-2]:
      self.mortality_event(t)
    #neworder.log("TOD=%f" % self.time_mortality)

  def mortality_event(self, t):
    self.alive = False
    self.time_mortality = self.age + t 
    # Person.__del__(self)

class People():
  """ A simple aggregration of Person """
  def __init__(self, mortality_hazard, n):
    # initialise population
    self.population = [ Person(mortality_hazard) for _ in range(n) ]
    self.life_expectancy = 0.0

  def age(self):
    [p.inc_age() for p in self.population]

  def calc_life_expectancy(self):  
    # compute mean
    le = 0.0
    n = 0
    for p in self.population:
      if not p.alive:
        le = le + p.time_mortality
        n = n + 1
    return le / n
    #self.life_expectancy = sum([p.time_mortality for p in self.population]) / len(self.population)
    #return self.life_expectancy

  def prop_alive(self):  
    # # compute mean
    neworder.log(sum([p.alive for p in self.population]) / len(self.population))
    return True

  # def fraction_alive(self, t):
  #   count = sum([p.state(t) for p in self.population]) / len(self.population)
