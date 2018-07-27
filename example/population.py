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
  def __init__(self, inputdata, asfr, asmr):
    self.data = pd.read_csv(inputdata)

    # Reformatting of input data is required to match Ethpop categories
    # actual age is randomised within the bound of the category
    self.data["Age"] = self.data.DC1117EW_C_AGE - np.random.uniform(size=len(self.data))
    self.data = _map_eth(self.data)

    # TODO might need to drop Sex column before unstack
    self.fertility = pd.read_csv(asfr, index_col=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"]) 
    self.mortality = pd.read_csv(asmr, index_col=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])

  def age(self, deltat):
    # TODO neworder log
    #print("[py] age", deltat)
    self.data.Age = self.data.Age + deltat
    # reconstruct census age group
    self.data.DC1117EW_C_AGE = np.clip(np.ceil(self.data.Age), 1, 86)

  def births(self, deltat):
    #print("[py] births", deltat)
    # First filter females
    females = self.data[self.data.DC1117EW_C_SEX == 2].copy()
    # might be a more efficient way of generating this array
    rates = females.join(self.fertility, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()
    # neworder callback 
    h = np.array(neworder.hazard_v(neworder.dvector.fromlist(rates * deltat)).tolist())
    # clone mothers, reset age and randomise gender
    newborns = females[h == 1].copy()
    newborns.Age = np.random.uniform(size=len(newborns)) # born within the last 12 months
    newborns.DC1117EW_C_AGE = 1 # this is 0-1 in census category
    # NOTE: do not convert to pd.Series here to stay as this has its own index which conflicts with the main table
    newborns.DC1117EW_C_SEX = np.array(neworder.hazard(0.5, len(newborns)).tolist()) + 1
    # append newborns to main population
    self.data = self.data.append(newborns)
  
  def migrations(self, deltat):
    pass

  def deaths(self, deltat):
    #print("[py] deaths", deltat)

    # might be a more efficient way of generating this array
    rates = self.data.join(self.mortality, on=["NewEthpop_ETH", "DC1117EW_C_SEX", "DC1117EW_C_AGE"])["Rate"].tolist()
    #print(rates)
    # neworder callback (requires inefficient conversions: Series/np.array -> list -> dvector -> list -> np.array)
    h = np.array(neworder.hazard_v(neworder.dvector.fromlist(rates * deltat)).tolist())
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
    return True # Faith

  def write_table(self, filename):
    self.data.to_csv(filename, index=False)
