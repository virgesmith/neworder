"""
Competing risks - fertility & mortality
"""
import numpy as np
import neworder

neworder.MAX_AGE = 100.0

# This is case-based model no timeline is required...
# ... but we do need a delta-t between entries in fertility/mortality data
neworder.timestep = 1.0

# Choose a simple linearly increasing mortality rate: 0.1% aged 0 to 2.5% aged 100
fertility_hazard_file = "examples/shared/NewETHPOP_fertility.csv"
mortality_hazard_file = "examples/shared/NewETHPOP_mortality.csv"
population_size = 100000
lad = "E09000030"
ethnicity = "WBI"

# running/debug options
neworder.log_level = 1
neworder.do_checks = False
 
# initialisation, this creates the population but doesnt assign a time of death
neworder.initialisations = {
  # a more efficient expression of the problem usin g pandas, runs about 5 times faster
  "people": { "module": "people", "class_": "People", "args": (fertility_hazard_file, mortality_hazard_file, lad, ethnicity, population_size) }
}

# transitions: simply samples time of death for each individual
neworder.transitions = {
  "age" : "people.age()",
}

neworder.checkpoints = {
  "stats": "people.stats()",
  "plot": "people.plot('./population.csv')"
  #"shell": "shell()" # uncomment for debugging
}
