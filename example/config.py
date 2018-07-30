
""" config.py
Microsimulation config
"""
import glob
import neworder

# define some global variables
initial_population = glob.glob("example/ssm_E0*_MSOA11_ppp_2011.csv")
#initial_population = "example/ssm_E08000021_MSOA11_2011.csv"
asfr = "example/TowerHamletsFertility.csv"
asmr = "example/TowerHamletsMortality.csv"

# running/debug options

# MPI prep - split initial population files over threads
threads = 3
def partition(arr, count):
  return [arr[i::count] for i in range(count)]
#print(split(initial_population, threads))

loglevel = 1
do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
checks = {
  "check": { "object": "people", "method": "check", "parameters" : [] }
  }
 
# initialisation
initialisations = {
  "people": { "module": "population", "class_": "Population", "parameters": [initial_population, asfr, asmr] }
}

# TODO need a mechanism to have deferred evaluation of parameters 
# e.g. for checkpoint data filename using current year

# define the evolution
neworder.timespan = neworder.DVector.fromlist([2011.25, 2020.25])
neworder.timestep = 0.5 # TODO beware rounding errors 
neworder.time = neworder.timespan[0]


# TODO timestep 
transitions = { 
  "fertility": { "object": "people", "method": "births", "parameters": [neworder.timestep] }, \
  "mortality": { "object": "people", "method": "deaths", "parameters": [neworder.timestep] }, \
  "age": { "object": "people", "method": "age", "parameters": [neworder.timestep] } \
  }

# Finalisation - YYYY gets replace with simulation time
output_file_pattern = "example/dm_YYYY.csv"
# TODO link to module when multiple
finalisations = {
  # "object": "people" # TODO link to module when multiple
  "write_table" : { "object": "people", "method": "write_table", "parameters": [output_file_pattern] }
}
