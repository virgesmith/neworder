
import collections
#import pandas as pd

Person = collections.namedtuple('Person', ['id', 'location', 'age', 'gender', 'ethnicity'])

class Population:
  def __init__(self):
    self.data = [ Person(id=0, age=30, gender='male', location="E09000001", ethnicity=1), \
                  Person(id=1, age=29, gender='female', location="", ethnicity="BLA")]

  def size(self):
    return len(self.data)

population = Population()

print("[python]", population.data[0])

# TODO directly call methods...
def get_size():
  return population.size()

def die():
  population.data.pop()

def birth():
  population.data.push(Person(id=3, age=0, gender="female", location="hosp", ethnicity="BLA"))
