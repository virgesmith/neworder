"""
Chapter 1
This is a direct neworder cover version of the Basic Cohort Model from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""

import neworder
import person

# "An arbitrarily selected value, chosen to produce a life expectancy of about 70 years."
neworder.mortality_hazard = 0.014
population_size = 10000

# running/debug options
loglevel = 1
do_checks = True
checks = { 
#  "life_expectancy": "people.life_expectancy >"
}
 
# initialisation, this creates the population but doesnt assign a time of death
initialisations = {
  "people": { "module": "person", "class_": "People", "parameters": [population_size] }
}

# use a large positive number to denote an infinite timespan (better than say -1 as it just works in inequalities)
neworder.time_infinity = 1e9
# This is case-based model - only a dummy timeline is required
neworder.timespan = neworder.DVector.fromlist([0, neworder.time_infinity])
neworder.timestep = neworder.time_infinity

# transitions: simply samples time of death for each individual
transitions = {
#  "tod" : "people.sample()"
  "tod" : "people.calc_life_expectancy()"
}

checkpoints = {
  "life_expectancy": "log(people.life_expectancy)",
  "shell": "shell()"
}
