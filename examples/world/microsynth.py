"""
Large-scale version of population.py
"""

import pandas as pd
import numpy as np
import humanleague
import neworder

import ethpop
from helpers import *

class Microsynth:
  def __init__(self, countries):

    country_lookup = pd.read_csv("./examples/world/data/CountryLookup.csv", sep="\t").set_index("Code")["Country"].to_dict()
    self.value_column = "2019 [YR2019]"

    self.countries = countries

    alldata = pd.read_csv("./examples/world/data/CountryData.csv").replace("..","")
    alldata[self.value_column] = pd.to_numeric(alldata[self.value_column]) 
    for country in self.countries:
      neworder.log("Microsynthesising population for %s" % country_lookup[country])
      data = alldata[(alldata["Country Code"] == country) & (alldata["Series Code"]).str.match("SP.POP.*(FE|MA)$")]
      # fallback to gender totals if age-specific values not available
      if data[self.value_column].isnull().values.any():
        neworder.log("%s: age-gender specific population data not available" % country_lookup[country])
        data = alldata[(alldata["Country Code"] == country) & (alldata["Series Code"]).str.match("^SP.POP.TOTL.(FE|MA).IN$")]
        # fallback to overall total if gender-specific values not available
        if data[self.value_column].isnull().values.any():
          neworder.log("%s: gender specific population data not available" % country_lookup[country])
          data = alldata[(alldata["Country Code"] == country) & (alldata["Series Code"]).str.match("^SP.POP.TOTL$")]
      else:
        data = pd.concat([data, data["Series Code"].str.split(".", expand=True)], axis=1) \
          .drop(["Country Code", "Series Code", 0, 1], axis=1) \
          .set_index([2,3]).unstack()
        pop = self.generate(data.values)

      neworder.log(data.head())

  def generate(self, agg_data):
    agg_data = humanleague.integerise(agg_data)["result"]
    # split 5y groups
    split = humanleague.prob2IntFreq(np.ones(5) / 5, int(agg_data.sum()))["freq"]

    pop = humanleague.qis([np.array([0,1], dtype=int), np.array([2], dtype=int)], [agg_data, split])

    if not isinstance(pop, dict):
      raise RuntimeError("microsynthesis general failure: %s" % pop)
    if not pop["conv"]:
      raise RuntimeError("microsynthesis convergence failure: %s" % pop)

    #neworder.log(pop["result"])
    raw = humanleague.flatten(pop["result"])#, np.array(["AGE5", "SEX", "AGE1"]))
    pop = pd.DataFrame(columns=["AGE5", "AGE1", "SEX"])
    pop.AGE5 = raw[0] 
    pop.AGE1 = raw[2] 
    pop.SEX = raw[1] 

  def write_table(self):
    # TODO define path in config
    filename = "./examples/world/data/pop2019_{}-{}.csv".format(neworder.rank(), neworder.size())
    neworder.log("writing %s" % filename)
    return self.data.to_csv(filename, index=False)
