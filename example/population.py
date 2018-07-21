"""
 population.py
"""

import pandas as pd

class Population:
  def __init__(self, inputdata):
    self.data = pd.read_csv(inputdata)

  def age(self, deltat):
    #print("[py] age(", deltat,")")
    self.data.DC1117EW_C_AGE = self.data.DC1117EW_C_AGE + deltat

  #def births(self):
    # neworder callback 

  #def deaths(self):
    # neworder callback 

  def mean_age(self):
    return self.data.DC1117EW_C_AGE.mean() - 1.0

  def size(self):
    return len(self.data)

  def finish(self, filename):
    self.data.to_csv(filename)
