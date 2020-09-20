"""
This is based on the model in Chapter 2, "The Life Table" from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""
import neworder
from people import People, People2
from plot import plot
import numpy as np
import time

#neworder.verbose()
neworder.checked(False)

max_age = 100.0

# Get some mortality rate data
mortality_hazard_file = "examples/shared/NewETHPOP_mortality.csv"
population_size = 100000

neworder.log("Population = %d" % population_size)

mortality = People(mortality_hazard_file, population_size, max_age)

start = time.perf_counter()
neworder.run(mortality)
end = time.perf_counter()
neworder.log("Discrete model life expectancy = %f, exec time = %f" % (mortality.calc_life_expectancy(), end - start))

mortality2 = People2(mortality_hazard_file, population_size, max_age)

start = time.perf_counter()
neworder.run(mortality2)
end = time.perf_counter()
neworder.log("Continuous model life expectancy = %f, exec time = %f" % (mortality2.calc_life_expectancy(), end - start))

plot(mortality.population, mortality2.population)
