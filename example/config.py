
""" config.py
Microsimulation config
"""
import glob
import neworder

# define some global variables
initial_populations = glob.glob("example/ssm_*_MSOA11_ppp_2011.csv")
#initial_population = "example/ssm_E08000021_MSOA11_2011.csv"
asfr = "example/TowerHamletsFertility.csv"
asmr = "example/TowerHamletsMortality.csv"

# running/debug options

# MP split initial population files over threads
def partition(arr, count):
  return [arr[i::count] for i in range(count)]

initial_populations = partition(initial_populations, neworder.nprocs)

print("[py] {}/{}:".format(neworder.procid, neworder.nprocs), initial_populations[neworder.procid])

#initial_population_array = split(initial_population, neworder.procid, neworder.nprocs))

loglevel = 1
do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
checks = {
  "check": { "object": "people", "method": "check", "parameters" : [] }
}
 
# initialisation
initialisations = {
  # TODO initial_populations[neworder.procid]
  "people": { "module": "population", "class_": "Population", "parameters": [initial_populations[neworder.procid], asfr, asmr] }
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
output_file_callback = neworder.Callback( '"example/dm_T_N_M.csv".replace("T_N_M", "{:.3f}_{}_{}".format(neworder.time, neworder.procid, neworder.nprocs))' )

# Finalisation 
# TODO rename to e.g. checkpoints
finalisations = {
  # "object": "people" # TODO link to module when multiple
  "write_table" : { "object": "people", "method": "write_table", "parameters": [output_file_callback] }
}
