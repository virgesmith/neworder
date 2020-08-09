"""
Chapter 1
This is based on the model in Chapter 2, "The Life Table" from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""
import numpy as np
import neworder
import person

max_age = 100.0

# This is case-based model - only a dummy timeline is required?
timeline = neworder.Timeline(0.0, max_age, [int(max_age)])

# Choose a simple linearly increasing mortality rate: 0.1% aged 0 to 2.5% aged 100
mortality_hazard_file = "examples/shared/NewETHPOP_mortality.csv"
population_size = 10000

from people import People
 
# running/debug options
checks = { 
  "alive": "neworder.model.prop_alive()"
}

# initialisation, this creates the population but doesnt assign a time of death
initialisations = {
  # nothing to initialise (other than the model itself)
}

# transitions: simply samples time of death for each individual
transitions = {
  "age" : "neworder.model.age()",
}

checkpoints = {
  "life_expectancy": "neworder.log(neworder.model.calc_life_expectancy())",
  "plot": "neworder.model.plot()"
}

neworder.model = People(timeline, initialisations, transitions, checks, checkpoints, mortality_hazard_file, population_size, max_age)
