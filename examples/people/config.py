
""" config.py
Microsimulation config
"""
import neworder

# define some global variables
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
loglevel = 1
# this model isnt meant for parallel execution
assert neworder.nprocs == 1, "This example is configured to be run as a single process only"

# initialisation
initialisations = {
  "people": { "module": "population", "class_": "Population", "parameters": [initial_population, asfr, asmr, asir, asor, ascr, asxr] }
}

# define the evolution
neworder.timespan = neworder.DVector.fromlist([2011.25, 2050.25])
neworder.timestep = 1.0 # TODO beware rounding errors 

# timestep must be defined in neworder
transitions = { 
  "fertility": "people.births(timestep)", 
  "mortality": "people.deaths(timestep)", 
  "migration": "people.migrations(timestep)", 
  "age": "people.age(timestep)" 
}

# checks to perform after each timestep. Assumed to return a boolean 
do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
checks = {
  "check": "people.check()"
}

# Generate output at each checkpoint  
checkpoints = {
  "check_data" : "people.check()",
  "write_table" : "people.write_table()" 
}