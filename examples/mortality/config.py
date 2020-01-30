"""
Chapter 1
This is based on the model in Chapter 2, "The Life Table" from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""
import numpy as np
import neworder
import person

neworder.MAX_AGE = 100.0

# This is case-based model - only a dummy timeline is required?
neworder.timeline = (0.0, neworder.MAX_AGE, [int(neworder.MAX_AGE)])

# Choose a simple linearly increasing mortality rate: 0.1% aged 0 to 2.5% aged 100
mortality_hazard_file = "examples/shared/NewETHPOP_mortality.csv"
population_size = 10000

# running/debug options
neworder.log_level = 1
neworder.do_checks = True
neworder.checks = { 
  "alive": "people.prop_alive()"
}
 
# initialisation, this creates the population but doesnt assign a time of death
neworder.initialisations = {
  # the MODGEN-style implementation
  #"people": { "module": "person", "class_": "People", "args": [mortality_hazard_file, population_size] },
  # a more efficient expression of the problem usin g pandas, runs about 5 times faster
  "people": { "module": "people", "class_": "People", "args": (mortality_hazard_file, population_size) }
}

# transitions: simply samples time of death for each individual
neworder.transitions = {
  "age" : "people.age()",
}

neworder.checkpoints = {
  "life_expectancy": "log(people.calc_life_expectancy())",
  "plot": "people.plot()"
  #"shell": "shell()" # uncomment for debugging
}
