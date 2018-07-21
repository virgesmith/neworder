# population.py

import pandas as pd

class Population:
  def __init__(self, inputdata):
    #inputdata = "src/test/ssm_E09000001_MSOA11_ppp_2011.csv"
    self.data = pd.read_csv(inputdata)

  def age(self, deltat):
    #print("[py] age(", deltat,")")
    self.data.DC1117EW_C_AGE = self.data.DC1117EW_C_AGE + deltat

  def mean_age(self):
    return self.data.DC1117EW_C_AGE.mean() - 1.0

  def finish(self, filename):
    self.data.to_csv(filename)
