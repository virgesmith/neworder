
""" config.py
Microsimulation config
"""
import glob

# define some global variables
initial_population = glob.glob("example/ssm_*_MSOA11_ppp_2011.csv")
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
# TODO link to module when multiple
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
timespan = [2011, 2020]
timestep = 1 # TODO breaks when not 1 
transitions = { 
  "fertility": { "object": "people", "method": "births", "parameters": [timestep] }, \
  "mortality": { "object": "people", "method": "deaths", "parameters": [timestep] }, \
  "age": { "object": "people", "method": "age", "parameters": [timestep] } \
  }

# Finalisation
final_population = "example/dm_" + str(timespan[-1]) + ".csv"
# TODO link to module when multiple
finalisations = {
  # "object": "people" # TODO link to module when multiple
  "write_table" : { "object": "people", "method": "write_table", "parameters": [final_population] }
}
