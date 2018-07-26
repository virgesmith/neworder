"""
 population.py
"""

import pandas as pd
import numpy as np
# TODO stub module
import neworder


def _map_eth(data):
  """ Maps census categories (DC2101EW_C_ETHPUK11) to NewEthpop. Note this is a one-way mapping """
  eth_map = { 0: "INV",
              1: "INV", 
              2: "WBI",
              3: "WHO", 
              4: "WHO",
              5: "WHO", 
              6: "MIX",
              7: "MIX",
              8: "MIX",
              9: "MIX",
              10: "MIX",
              11: "INV",
              12: "IND",
              13: "PAK",
              14: "BAN",
              15: "CHI",
              16: "OAS",
              17: "INV",
              18: "BLA",
              19: "BLC",
              20: "OBL",
              21: "OTH",
              22: "OTH", 
              23: "OTH" } 
  data["NewEthpop_ETH"] = data.DC2101EW_C_ETHPUK11.map(eth_map) #, na_action=None)
  return data.drop("DC2101EW_C_ETHPUK11", axis=1)
 
class Population:
  def __init__(self, inputdata):
    self.data = pd.read_csv(inputdata)

    # Reformatting of input data is required to match Ethpop categories
    # actual age is randomised
    self.data["Age"] = self.data.DC1117EW_C_AGE - np.random.uniform(size=len(self.data))

    self.data = _map_eth(self.data)

    # TODO might need to drop Sex column before unstack
    fertility = pd.read_csv("./example/TowerHamletsFertility.csv", index_col=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"]) #.unstack()
    mortality = pd.read_csv("./example/TowerHamletsMortality.csv", index_col=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"]) #.unstack()

    print(fertility.head())
    print(mortality.head())
    print(self.data.head())

  def age(self, deltat):
    # TODO neworder log
    #print("[py] age", deltat)
    self.data.Age = self.data.Age + deltat
    # reconstruct census age group
    self.data.DC1117EW_C_AGE = np.clip(np.ceil(self.data.Age), 1, 86)

  def births(self, deltat, rate):
    #print("[py] births", deltat, rate)
    # neworder callback 
    # First filter females
    females = self.data[self.data.DC1117EW_C_SEX == 2].copy()
    h = np.array(neworder.hazard(rate * deltat, len(females)).tolist()) 
    # clone mothers, reset age and randomise gender
    newborns = females[h == 1].copy()
    newborns.Age = np.random.uniform(size=len(newborns))
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
    return self.data.Age.mean()

  def gender_split(self):
    # this is % female
    return self.data.DC1117EW_C_SEX.mean() - 1.0

  def size(self):
    return len(self.data)

  def check(self):
    print("[py] check OK: size={} mean_age={:.2f}, pct_female={:.2f}".format(self.size(), self.mean_age(), 100.0 * self.gender_split()))
    return True

  def finish(self, filename):
    self.data.to_csv(filename, index=False)
