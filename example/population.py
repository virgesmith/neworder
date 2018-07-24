"""
 population.py
"""

import pandas as pd

import neworder

class Population:
  def __init__(self, inputdata):
    self.data = pd.read_csv(inputdata)

  def age(self, deltat):
    #print("[py] age(", deltat,")")
    self.data.DC1117EW_C_AGE = self.data.DC1117EW_C_AGE + deltat

  #def births(self):
    # neworder callback 

  def deaths(self, rate):
    # neworder callback
    h = neworder.hazard(rate, len(self.data)) 
    self.data["DEAD"] = h.tolist()
    # remove deceased
    self.data = self.data[self.data.DEAD == 0]
    # remove temp column
    self.data.drop(["DEAD"], axis=1, inplace=True)
    
  def mean_age(self):
    return self.data.DC1117EW_C_AGE.mean() - 1.0

  def size(self):
    return len(self.data)

  def finish(self, filename):
    self.data.to_csv(filename)
