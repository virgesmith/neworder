"""
RiskPaths
This will be a neworder implementation of the RiskPaths MODGEN model

See: 
https://www.statcan.gc.ca/eng/microsimulation/modgen/new/chap3/chap3
https://www.statcan.gc.ca/eng/microsimulation/modgen/new/chap4/chap4

  'RiskPaths is a simple, competing risk, case-based continuous time microsimulation model. Its
  main use is as a teaching tool, introducing microsimulation to social scientists and demonstrating
  how dynamic microsimulation models can be efficiently programmed using the language
  Modgen.
  Modgen is a generic microsimulation programming language developed and maintained at
  Statistics Canada.
  RiskPaths as well as the Modgen programming language and other related documents are
  available at www.statcan.gc.ca/microsimulation/modgen/modgen-eng.htm'

"""
import numpy as np
import neworder

# TODO parameterise
neworder.mortality_rate = 0.01
neworder.fertility_rate = 0.01

population_size = 100

# running/debug options
log_level = 1
do_checks = True # Faith
# assumed to be methods of class_ returning True if checks pass
checks = {
  #"check": "people.check()"
}
 
# initialisation
initialisations = {
  "people": { "module": "riskpaths", "class_": "RiskPaths", "parameters": [population_size] }
}

# This timeline represents persons age
# range LIFE
neworder.timespan = np.array([0, 100])
neworder.timestep = 1


transitions = {
  "status": "people.alive()"
}


# Finalisation 
checkpoints = {
}
