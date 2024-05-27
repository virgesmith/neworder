"""
Chapter 1
This is a direct neworder cover version of the Basic Cohort Model from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""
import numpy as np
from person import People

import neworder

# neworder.verbose() # uncomment for detailed output

# "An arbitrarily selected value, chosen to produce a life expectancy of about 70 years."
# (see the "mortality" example for a more realistic model)
mortality_hazard = 0.014
population_size = 100000

model = People(mortality_hazard, population_size)

neworder.run(model)

# now we can sample the population generated by the model to see the proportion of deaths at (arbitrarily) 10 year intervals
for age in np.linspace(10.0, 100.0, 10):
    neworder.log("Age %.0f survival rate = %.1f%%" % (age, model.alive(age) * 100.0))
