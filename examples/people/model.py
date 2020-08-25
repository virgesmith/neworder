
""" config.py
Microsimulation config
"""
import neworder

from population import Population


# define some global variables describing where the starting population and the parameters of the dynamics come from
initial_population = "examples/people/ssm_E08000021_MSOA11_ppp_2011.csv"
asfr = "examples/shared/NewETHPOP_fertility.csv"
asmr = "examples/shared/NewETHPOP_mortality.csv"
# internal in-migration
asir = "examples/shared/NewETHPOP_inmig.csv"
# internal out-migration
asor = "examples/shared/NewETHPOP_outmig.csv"
# immigration
ascr = "examples/shared/NewETHPOP_immig.csv"
# emigration
asxr = "examples/shared/NewETHPOP_emig.csv"

# this model isnt meant for parallel execution
assert neworder.mpi.size() == 1, "This example is configured to be run as a single process only"

# define the evolution timeline
timeline = neworder.Timeline(2011, 2021, [7,10])

population = Population(timeline, initial_population, asfr, asmr, asir, asor, ascr, asxr)

neworder.run(population)
