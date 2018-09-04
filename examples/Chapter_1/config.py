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
population_size = 10000

# running/debug options
neworder.log_level = 1
neworder.do_checks = True
neworder.checks = { 
#  "life_expectancy": "people.life_expectancy >"
}
 
# initialisation, this creates the population but doesnt assign a time of death
neworder.initialisations = {
  "people": { "module": "person", "class_": "People", "parameters": [mortality_hazard, population_size] }
}

# use a large positive number to denote an infinite timespan (better than say -1 as it just works in inequalities)
TIME_INFINITY = 1e9
# This is case-based model - only a dummy timeline is required?
neworder.timespan = np.array([0, TIME_INFINITY])
neworder.timestep = TIME_INFINITY

# transitions: simply samples time of death for each individual
neworder.transitions = {
#  "tod" : "people.sample()"
  "tod" : "people.calc_life_expectancy()"
}

neworder.checkpoints = {
  "life_expectancy": "log(people.life_expectancy)",
  #"shell": "shell()" # uncomment for debugging
}
