
"""
model.py: Population Microsimulation - births, deaths and migration by age, gender and ethnicity
"""
import time
from datetime import date

import neworder
from population import Population

neworder.verbose()

# input data
initial_population = "examples/people/E08000021_MSOA11_2011.csv"
# age, gender and ethnicity-specific rates
fertility_rate_data = "examples/people/ethpop_fertility.csv"
mortality_rate_data = "examples/people/ethpop_mortality.csv"
# TODO the numbers aren't being interpreted correctly
in_migration_rate_data = "examples/people/ethpop_inmig.csv"
out_migration_rate_data = "examples/people/ethpop_outmig.csv"

# define the evolution timeline
timeline = neworder.CalendarTimeline(date(2011, 1, 1), date(2051, 1, 1), 1, "y", 4)

# create the model
population = Population(timeline, initial_population, fertility_rate_data, mortality_rate_data, in_migration_rate_data, out_migration_rate_data)

# run the model
start = time.time()
ok = neworder.run(population)
neworder.log("run time = %.2fs" % (time.time() - start))
assert ok



