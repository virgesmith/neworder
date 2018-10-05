
""" config.py
Microsimulation config
"""
import numpy as np
import neworder

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

# running/debug options
neworder.log_level = 1
# this model isnt meant for parallel execution
assert neworder.size() == 1, "This example is configured to be run as a single process only"

# define the outer sequence loop (optional)
# run 4 sims
neworder.sequence = np.array([3,1,2,0])
# define the evolution
neworder.timeline = (2011, 2018, 2021, 10)

# initialisation
neworder.initialisations = {
  "people": { "module": "population", "class_": "Population", "parameters": [initial_population, asfr, asmr, asir, asor, ascr, asxr] }
}

# timestep must be defined in neworder
neworder.transitions = { 
  "2fertility": "people.births(timestep)", 
  "1mortality": "people.deaths(timestep)", 
  "3migration": "people.migrations(timestep)", 
  "4age": "people.age(timestep)" 
}

# checks to perform after each timestep. Assumed to return a boolean 
neworder.do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
neworder.checks = {
  "check": "people.check()"
}

# Generate output at each checkpoint  
neworder.checkpoints = {
#  "check_data" : "people.check()",
  "write_table" : "people.write_table()" 
}