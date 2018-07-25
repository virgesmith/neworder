"""
 population.py
"""

import pandas as pd
import numpy as np
import neworder

class Population:
  def __init__(self, inputdata):
    self.data = pd.read_csv(inputdata)

  def age(self, deltat):
    # TODO neworder log
    #print("[py] age", deltat)
    self.data.DC1117EW_C_AGE = self.data.DC1117EW_C_AGE + deltat

  def births(self, deltat, rate):
    #print("[py] births", deltat, rate)
    # neworder callback 
    # First filter females
    females = self.data[self.data.DC1117EW_C_SEX == 2].copy()
    h = neworder.hazard(rate * deltat, len(females)) 
    # TODO do we actually need to append this column???
    females["BORN"] = h.tolist()
    # clone mothers, reset age and randomise gender
    newborns = females[females.BORN == 1].copy()
    newborns.DC1117EW_C_AGE = 1
    newborns.DC1117EW_C_SEX = np.random.choice([1,2])
    # remove temp column
    newborns = newborns.drop(["BORN"], axis=1)
    self.data = self.data.append(newborns)

  def deaths(self, deltat, rate):
    #print("[py] deaths", deltat, rate)
    # neworder callback
    h = neworder.hazard(rate * deltat, len(self.data)) 
    self.data["DEAD"] = h.tolist()
    # remove deceased
    self.data = self.data[self.data.DEAD == 0]
    # remove temp column
    self.data.drop(["DEAD"], axis=1, inplace=True)
    
  def mean_age(self):
    return self.data.DC1117EW_C_AGE.mean() - 1.0

  def gender_split(self):
    # this is % female
    return self.data.DC1117EW_C_SEX.mean() - 1.0

  def size(self):
    return len(self.data)

  def finish(self, filename):
    self.data.to_csv(filename)
