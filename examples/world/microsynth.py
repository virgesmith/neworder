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
    self.countries = countries

    neworder.log("Microsynthesising populations for %s" % ", ".join(self.countries))

    alldata = pd.read_csv("./examples/world/data/CountryData.csv")
    # SP.POP.0004.FE
    # ...
    # SP.POP.80UP.FE
    # SP.POP.0004.MA
    # ...
    # SP.POP.80UP.MA
    for country in self.countries:
      data = alldata[(alldata["Country Code"] == country) & (alldata["Series Code"]).str.match("SP.POP.*[FE|MA]$")]
      neworder.log(data.head())
      #neworder.log("%s pop = %d" % (data[(data.Country == country) && ()]))


  def write_table(self):
    # TODO define path in config
    filename = "./examples/world/data/pop2019_{}-{}.csv".format(neworder.rank(), neworder.size())
    neworder.log("writing %s" % filename)
    return self.data.to_csv(filename, index=False)
