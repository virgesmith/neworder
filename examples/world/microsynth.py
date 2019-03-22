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

    value_column = "2019 [YR2019]"
    self.countries = countries

    neworder.log("Microsynthesising populations for %s" % ", ".join(self.countries))

    alldata = pd.read_csv("./examples/world/data/CountryData.csv").replace("..","")
    alldata[value_column] = pd.to_numeric(alldata[value_column]) 
    # SP.POP.0004.FE
    # ...
    # SP.POP.80UP.FE
    # SP.POP.0004.MA
    # ...
    # SP.POP.80UP.MA
    for country in self.countries:
      data = alldata[(alldata["Country Code"] == country) & (alldata["Series Code"]).str.match("SP.POP.*(FE|MA)$")]
      # fallback to totals if age-speicific values not available
      if data[value_column].isnull().values.any():
        data = alldata[(alldata["Country Code"] == country) & (alldata["Series Code"]).str.match("^SP.POP.TOTL$")]
      
      neworder.log(data)
      #neworder.log("%s pop = %d" % (data[(data.Country == country) && ()]))


  def write_table(self):
    # TODO define path in config
    filename = "./examples/world/data/pop2019_{}-{}.csv".format(neworder.rank(), neworder.size())
    neworder.log("writing %s" % filename)
    return self.data.to_csv(filename, index=False)
