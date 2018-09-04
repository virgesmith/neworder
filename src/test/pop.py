
DEPRECATED

import collections
#import numpy as np
import pandas as pd
import neworder as no


Person = collections.namedtuple('Person', ['id', 'location', 'age', 'gender', 'ethnicity'])

class Population:
  def __init__(self):
    self.data = [ Person(id=0, age=30, gender='male', location="E09000001", ethnicity=1), \
                  Person(id=1, age=29, gender='female', location="", ethnicity="BLA")]

    #self.array = np.array([1,2,3,4,5,6])
    # not using numpy for now
    self.array = [1.0,2,3,4,5,6]

    self.data = pd.read_csv("../../examples/people/ssm_E09000001_MSOA11_ppp_2011.csv")

    self.double = 1.0

  def columns(self):
    # columns as np array
    return self.data.columns.values.tolist()

  def values(self):
    # data as np array 
    return self.data.values.tolist()

  def size(self):
    return len(self.array)

  def print(self):
    no.log("got {}, adding 10...".format(self.array)) 
    for i in range(0, len(self.array)):
      self.array[i] = self.array[i] + 10

population = Population()

def func():
  return population.data
