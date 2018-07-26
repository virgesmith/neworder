
# config.py

# define some global variables
initial_population = "example/ssm_E09000001_MSOA11_ppp_2011.csv"
mortality_hazard = 0.01
birth_rate = 0.02

# debug options
loglevel = 1
do_checks = True
# assumed to be methods of class_ returning True if checks pass
# TODO link to module when multiple
checks = {
  "check": { "method": "check", "parameters" : [] }
  }
 
# initialisation
# TODO multiple modules...
module = "population"
class_ = "Population"
parameters = initial_population # TODO workaround string splitting

# define the evolution
# TODO link to module when multiple
timespan = [2011, 2020]
timestep = 1
transitions = { 
  "fertility": { "method": "births", "parameters": [timestep, birth_rate] }, \
  "mortality": { "method": "deaths", "parameters": [timestep, mortality_hazard] }, \
  "age": { "method": "age", "parameters": [timestep] } \
  }

# Finalisation
final_population = initial_population.replace(str(timespan[0]), str(timespan[1]))
# TODO link to module when multiple
finalisations = {
  "finish" : { "method": "finish", "parameters": [final_population] }
}
