
""" config.py
Microsimulation config for mulit-LAD MPI simulation
"""
import numpy as np
import glob
import neworder

# define some global variables describing where the starting population and the parameters of the dynamics come from
initial_populations = glob.glob("examples/people_multi/data/ssm_*_MSOA11_ppp_2011.csv")
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

# MPI split initial population files over threads
def partition(arr, count):
  return [arr[i::count] for i in range(count)]

initial_populations = partition(initial_populations, neworder.nprocs)

# running/debug options
neworder.log_level = 1
 
# initialisation
neworder.initialisations = {
  "people": { "module": "population", "class_": "Population", "parameters": [initial_populations[neworder.procid], asfr, asmr, asir, asor, ascr, asxr] }
}

# define the evolution
neworder.timespan = np.array([2011.25, 2050.25])
neworder.timestep = 1.0 # TODO beware rounding errors 

# timestep must be defined in neworder
neworder.transitions = { 
  "fertility": "people.births(timestep)", 
  "mortality": "people.deaths(timestep)", 
  "migration": "people.migrations(timestep)", 
  "age": "people.age(timestep)" 
}

# checks to perform after each timestep. Assumed to return a boolean 
neworder.do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
neworder.checks = {
  "check": "people.check()"
}

# Generate output at each checkpoint  
neworder.checkpoints = {
  #"check_data" : "people.check()",
  "write_table" : "people.write_table()"  
}
