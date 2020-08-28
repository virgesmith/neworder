"""
This is based on the model in Chapter 2, "The Life Table" from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""
import neworder
from people import People
from plot import plot

#neworder.verbose()

max_age = 100.0

# Get some mortality rate data
mortality_hazard_file = "examples/shared/NewETHPOP_mortality.csv"
population_size = 100000

mortality = People(mortality_hazard_file, population_size, max_age)

neworder.run(mortality)

plot(mortality.population)
