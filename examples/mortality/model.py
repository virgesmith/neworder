"""
This is based on the model in Chapter 2, "The Life Table" from the Belanger & Sabourin book
See https://www.microsimulationandpopulationdynamics.com/
"""
import neworder
from people import PeopleDiscrete, PeopleContinuous
import numpy as np
import time
from plot import plot

#neworder.verbose()
# checks disabled to emphasis performance differences
neworder.checked(False)

max_age = 100.0

# Get some mortality rate data
mortality_hazard_file = "examples/mortality/ethpop_mortality_wbi.csv"
population_size = 100000

neworder.log("Population = %d" % population_size)

mortality_discrete = PeopleDiscrete(mortality_hazard_file, population_size, max_age)

start = time.perf_counter()
neworder.run(mortality_discrete)
end = time.perf_counter()
neworder.log("Discrete model life expectancy = %f, exec time = %f" % (mortality_discrete.calc_life_expectancy(), end - start))

mortality_continuous = PeopleContinuous(mortality_hazard_file, population_size, max_age)

start = time.perf_counter()
neworder.run(mortality_continuous)
end = time.perf_counter()
neworder.log("Continuous model life expectancy = %f, exec time = %f" % (mortality_continuous.calc_life_expectancy(), end - start))

plot(mortality_discrete.population, mortality_continuous.population)
