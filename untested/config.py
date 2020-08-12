"""
Chapter 1
This is a direct neworder cover version of the Basic Cohort Model from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""
import numpy as np
import neworder
import person

# "An arbitrarily selected value, chosen to produce a life expectancy of about 70 years."
mortality_hazard = 0.014
population_size = 100000

# running/debug options
neworder.log_level = 1
neworder.do_checks = False
 
# initialisation, this creates the population but doesnt assign a time of death
neworder.initialisations = {
  "people": { "module": "person", "class_": "People", "args": (mortality_hazard, population_size) }
}

# This is case-based model with no explicit time evolution 
# no need to set a timeline: internally this means we just have one dummy timestep

# transitions: simply samples time of death for each individual
neworder.dataframe.transitions = {
  "time_of_death" : "people.sample_mortality()"
}

neworder.checkpoints = {
  "life_expectancy": "people.calc_life_expectancy()",
}
