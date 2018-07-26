"""
 population.py
"""

import pandas as pd
import numpy as np
# TODO stub module
import neworder

class Population:
  def __init__(self, inputdata):
    self.data = pd.read_csv(inputdata)
    # TODO might need to drop Sex column before unstack
    fertility = pd.read_csv("./example/TowerHamletsFertility.csv", sep=";", index_col=["Ethnicity", "Sex", "Age"]) #.unstack()
    mortality = pd.read_csv("./example/TowerHamletsMortality.csv", sep=";", index_col=["Ethnicity", "Sex", "Age"]) #.unstack()
    #print(mortality.shape)

  def age(self, deltat):
    # TODO neworder log
    #print("[py] age", deltat)
    self.data.DC1117EW_C_AGE = self.data.DC1117EW_C_AGE + deltat

  def births(self, deltat, rate):
    #print("[py] births", deltat, rate)
    # neworder callback 
    # First filter females
    females = self.data[self.data.DC1117EW_C_SEX == 2].copy()
    h = np.array(neworder.hazard(rate * deltat, len(females)).tolist()) 
    # clone mothers, reset age and randomise gender
    newborns = females[h == 1].copy()
    newborns.DC1117EW_C_AGE = 1 # this is 0-1 in census category
    # NOTE: do not convert to pd.Series here as this has its own index which conflicts with the main table
    newborns.DC1117EW_C_SEX = np.array(neworder.hazard(0.5, len(newborns)).tolist()) + 1
    # this is non-deterministic...
    #newborns.DC1117EW_C_SEX = np.random.choice([1,2]) # this is not deterministic
    # append newborns to main population
    self.data = self.data.append(newborns)
  
#  def migrations(self, deltat, rate)

  def deaths(self, deltat, rate):
    #print("[py] deaths", deltat, rate)
    # neworder callback (requires conversion to series/np.array)
    h = np.array(neworder.hazard(rate * deltat, len(self.data)).tolist())
    # remove deceased
    self.data = self.data[h!=1]
    
  def mean_age(self):
    return self.data.DC1117EW_C_AGE.mean() - 1.0

  def gender_split(self):
    # this is % female
    return self.data.DC1117EW_C_SEX.mean() - 1.0

  def size(self):
    return len(self.data)

  def check(self):
    print("[py] check OK: size={} mean_age={:.2f}, pct_female={:.2f}".format(self.size(), self.mean_age(), 100.0 * self.gender_split()))
    return True

  def finish(self, filename):
    self.data.to_csv(filename)
