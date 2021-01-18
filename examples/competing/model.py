"""
Competing risks - fertility & mortality
"""
import neworder
# model implementation
from people import People
# separate visualisation code
from visualise import plot

#neworder.verbose()

# create model
# data are for white British women in a London Borough at 1 year time resolution
dt = 1.0 # years
fertility_hazard_file = "examples/competing/fertility-wbi.csv"
mortality_hazard_file = "examples/competing/mortality-wbi.csv"
population_size = 100000
pop = People(dt, fertility_hazard_file, mortality_hazard_file, population_size)

# run model
neworder.run(pop)

# visualise results
plot(pop)

