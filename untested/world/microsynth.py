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

    self.pop = pd.DataFrame()

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
          assert len(data) == 1
          if np.isnan(data[self.value_column].values):
            neworder.log("%s: total population data not available - skipping" % country)
          else:
            self._generate_from_total(data[self.value_column].values, country)
        else: 
          raise NotImplementedError("microsynth from M/F totals")
      else:
        data = pd.concat([data, data["Series Code"].str.split(".", expand=True)], axis=1) \
          .drop(["Country Code", "Series Code", 0, 1], axis=1) \
          .set_index([2,3]).unstack()
        # get synth pop for the country
        self._generate(data.values, country)

    # memory used in MB scaled up to world pop
    #neworder.log(self.pop.memory_usage() / len(self.pop) * 7.5e9 / 1024 / 1024)

  def _generate(self, agg_data, country):
    agg_data = humanleague.integerise(agg_data)["result"]
    # split 5y groups
    split = humanleague.prob2IntFreq(np.ones(5) / 5, int(agg_data.sum()))["freq"]

    msynth = humanleague.qis([np.array([0,1], dtype=int), np.array([2], dtype=int)], [agg_data, split])
    if not isinstance(msynth, dict):
      raise RuntimeError("microsynthesis general failure: %s" % msynth)
    if not msynth["conv"]:
      raise RuntimeError("microsynthesis convergence failure")

    #neworder.log(pop["result"])
    raw = humanleague.flatten(msynth["result"])
    pop = pd.DataFrame(columns=["AGE5", "AGE1", "SEX"])
    pop.AGE5 = raw[0] 
    pop.AGE1 = raw[2] 
    pop.SEX = raw[1]

    # could fail here if zero people in any category
    assert len(pop.AGE5.unique()) == 17
    assert len(pop.AGE1.unique()) == 5
    assert len(pop.SEX.unique()) == 2

    # construct single year of age 
    pop["Country"] = country
    pop["AGE"] = pop.AGE5 * 5 + pop.AGE1 
    self.pop = self.pop.append(pop.drop(["AGE5", "AGE1"], axis=1))

  def _generate_from_total(self, agg_value, country):
    # TODO improve distribution
    sex_split = humanleague.prob2IntFreq(np.ones(2) / 2, int(agg_value))["freq"]
    age_split = humanleague.prob2IntFreq(np.ones(17) / 17, int(agg_value))["freq"]

    msynth = humanleague.qis([np.array([0], dtype=int), np.array([1], dtype=int)], [age_split, sex_split])
    if not isinstance(msynth, dict):
      raise RuntimeError("microsynthesis (from total) general failure: %s" % msynth)
    if not msynth["conv"]:
      raise RuntimeError("microsynthesis (from total) convergence failure")

    raw = humanleague.flatten(msynth["result"])
    pop = pd.DataFrame(columns=["AGE", "SEX"])
    pop.AGE = raw[0] 
    pop.SEX = raw[1]

    # could fail here if zero people in any category
    assert len(pop.AGE.unique()) == 17
    assert len(pop.SEX.unique()) == 2

    # construct single year of age 
    pop["Country"] = country
    self.pop = self.pop.append(pop, sort=False)

  def write_table(self):
    # TODO define path in config
    filename = "./examples/world/data/pop2019_{}-{}.csv".format(neworder.mpi.rank(), neworder.mpi.size())
    neworder.log("writing %s" % filename)
    return self.pop.to_csv(filename, index=False)
