
""" config.py
Microsimulation config for World population microsimulation
Data soruced from the World Bank, https://databank.worldbank.org
"""
import numpy as np
import pandas as pd
import glob
import os
import neworder

# MPI split initial population files over threads
def partition(arr, count):
  if count > 1:
    return [arr[i::count] for i in range(count)]
  return [arr]

allcountries = pd.read_csv("./examples/world/data/CountryLookup.csv", encoding='utf-8', sep="\t")["Code"][:4]

initial_populations = partition(allcountries, neworder.size())

# running/debug options
neworder.log_level = 1
 
# initialisation
neworder.initialisations = {
  "people": { "module": "microsynth", "class_": "Microsynth", "parameters": [initial_populations[neworder.rank()]] }
}

# define the evolution
neworder.timeline = (2019, 2030, 12)

# timestep must be defined in neworder
neworder.transitions = { 
}

# checks to perform after each timestep. Assumed to return a boolean 
neworder.do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
neworder.checks = {
}

# Generate output at each checkpoint  
neworder.checkpoints = {
}
