
""" config.py
Microsimulation config
"""
import glob
import neworder

# define some global variables
initial_populations = glob.glob("examples/people/ssm_E08*_MSOA11_ppp_2011.csv")
asfr = "examples/people/TowerHamletsFertility.csv"
asmr = "examples/people/TowerHamletsMortality.csv"
asir = "examples/people/NewETHPOP_inmig.csv"
asor = "examples/people/NewETHPOP_outmig.csv"

# MP split initial population files over threads
def partition(arr, count):
  return [arr[i::count] for i in range(count)]

initial_populations = partition(initial_populations, neworder.nprocs)

# running/debug options
loglevel = 1
 
# initialisation
initialisations = {
  "people": { "module": "population", "class_": "Population", "parameters": [initial_populations[neworder.procid], asfr, asmr, asir, asor] }
}

# define the evolution
neworder.timespan = neworder.DVector.fromlist([2011.25, 2015.25, 2020.25])
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
  "write_table" : "people.write_table()" 
}