
""" config.py
Microsimulation config
"""
import glob
import neworder

# define some global variables
initial_population = glob.glob("example/ssm_E09*_MSOA11_ppp_2011.csv")
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

# mechanisms to have deferred/shared evaluation of parameters:
# 1) store a value in neworder 
# 2) construct a Callback to be evaluated as needed  

# define the evolution
neworder.timespan = neworder.DVector.fromlist([2011.25, 2015.25, 2020.25])
neworder.timestep = 1.0 # TODO beware rounding errors 


# TODO timestep 
transitions = { 
  "fertility": { "object": "people", "method": "births", "parameters": [neworder.timestep] }, \
  "mortality": { "object": "people", "method": "deaths", "parameters": [neworder.timestep] }, \
  "age": { "object": "people", "method": "age", "parameters": [neworder.timestep] } \
  }

# generates filename according to current time TODO and thread (MPI_COMM_RANK)
output_file_callback = neworder.Callback( '"example/dm_YYYY.csv".replace("YYYY", "{:.3f}".format(neworder.time))' )

# Finalisation 
# TODO rename to e.g. checkpoints
finalisations = {
  # "object": "people" # TODO link to module when multiple
  "write_table" : { "object": "people", "method": "write_table", "parameters": [output_file_callback] }
}
