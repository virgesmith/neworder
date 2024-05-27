import time

# model implementations
from people import PeopleContinuous, PeopleDiscrete

# visualisation code
from plot import plot

import neworder

# neworder.verbose()
# checks disabled to emphasise performance differences
neworder.checked(False)

# the max value in the timeline
max_age = 100.0
# The mortality rate data
mortality_hazard_file = "examples/mortality/mortality-wbi.csv"
population_size = 100000

neworder.log("Population = %d" % population_size)

# run the discrete model
mortality_discrete = PeopleDiscrete(mortality_hazard_file, population_size, max_age)
start = time.perf_counter()
neworder.run(mortality_discrete)
end = time.perf_counter()
neworder.log(
    "Discrete model life expectancy = %f, exec time = %f"
    % (mortality_discrete.life_expectancy, end - start)
)

# run the continuous model
mortality_continuous = PeopleContinuous(mortality_hazard_file, population_size, 1.0)
start = time.perf_counter()
neworder.run(mortality_continuous)
end = time.perf_counter()
neworder.log(
    "Continuous model life expectancy = %f, exec time = %f"
    % (mortality_continuous.life_expectancy, end - start)
)

# visualise some results
# hist_file = "docs/examples/img/mortality_%dk.png" % (population_size//1000)
# anim_file = "docs/examples/img/mortality_hist_%dk.gif" % (population_size//1000)
plot(mortality_discrete.population, mortality_continuous.population)
